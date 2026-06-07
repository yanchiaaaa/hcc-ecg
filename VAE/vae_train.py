import torch
from torch import nn
from torch.utils.data import DataLoader
import os, logging, copy
from tqdm import tqdm
import multiprocessing as mp
from VAE.vae_model import VAE_Decoder, VAE_Encoder, loss_function
from dataset.mimic_processed_dataset import MIMIC_IV_ECG_Processed_Dataset

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            esd = self.model.state_dict()
            for name, param in msd.items():
                if name in esd: esd[name].copy_(self.decay * esd[name] + (1.0 - self.decay) * param)


def val_loop(dataloader, encoder, decoder, loss_fn, kld_weight, device):
    encoder.eval(); decoder.eval()
    total_metrics = {'loss': 0, 'mse': 0, 'spectral': 0, 'KLD': 0}
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            if X.dim() == 4: X = X.view(-1, 1024, 12) 
            z, mean, log_var = encoder(X)
            loss_dict = loss_fn(decoder(z), X, mean, log_var, kld_weight)
            for k in total_metrics: 
                total_metrics[k] += loss_dict[k].item()
    n = len(dataloader)
    return [total_metrics[k]/n for k in ['loss', 'mse', 'spectral', 'KLD']]

def train_loop(dataloader, encoder, decoder, loss_fn, optimizer, scheduler, kld_weight, device, ema_encoder=None, ema_decoder=None, accumulation_steps=1, epoch_desc=""):
    encoder.train(); decoder.train()
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)
    total_metrics = {'loss': 0, 'mse': 0, 'spectral': 0, 'KLD': 0}

    for batch_idx, (X, _) in enumerate(progress_bar):
        X = X.to(device)
        if X.dim() == 4: X = X.view(-1, 1024, 12) 
        
        optimizer.zero_grad()
        z, mean, log_var = encoder(X)
        loss_dict = loss_fn(decoder(z), X, mean, log_var, kld_weight)
        
        loss = loss_dict['loss'] / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            if ema_encoder: ema_encoder.update(encoder)
            if ema_decoder: ema_decoder.update(decoder)

        for k in total_metrics: 
            total_metrics[k] += loss_dict[k].item()
            
        progress_bar.set_postfix({
            'MSE': f"{total_metrics['mse']/(batch_idx+1):.4f}", 
            'KLD': f"{total_metrics['KLD']/(batch_idx+1):.4f}"
        })
    n = len(dataloader)
    return [total_metrics[k]/n for k in ['loss', 'mse', 'spectral', 'KLD']]

if __name__ == '__main__':
    try: mp.set_start_method('spawn')
    except RuntimeError: pass

    save_path = "checkpoints/vae"
    os.makedirs(save_path, exist_ok=True)
    
    logger = logging.getLogger('vae_opt_v2'); logger.setLevel('INFO')
    fh = logging.FileHandler(os.path.join(save_path, 'train.log'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s'); fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)

    H_ = {'lr': 1e-4, 'batch_size': 256, 'epochs': 50, 'num_workers': 0, 'accumulation_steps': 1}
    logger.info(H_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    path = 'data/processed_data_icd'
    train_dl = DataLoader(MIMIC_IV_ECG_Processed_Dataset(path, 'train', True), H_['batch_size'], num_workers=H_['num_workers'], pin_memory=True, shuffle=True)
    val_dl = DataLoader(MIMIC_IV_ECG_Processed_Dataset(path, 'val', True), 256, num_workers=H_['num_workers'], pin_memory=True)

    encoder, decoder = VAE_Encoder().to(device), VAE_Decoder().to(device)
    ema_enc, ema_dec = EMA(encoder), EMA(decoder)
    if gpu_count > 1: encoder, decoder = nn.DataParallel(encoder), nn.DataParallel(decoder)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=H_['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=H_['epochs'], steps_per_epoch=len(train_dl)//H_['accumulation_steps'])
    
    best_val_loss = float('inf')
    
    for t in range(H_['epochs']):
        epoch_num = t + 1
        
        max_kld = 0.05
        curr_kld = (t / H_['epochs']) * max_kld
        
        t_metrics = train_loop(
            train_dl, encoder, decoder, loss_function, optimizer, scheduler, curr_kld, device, 
            ema_enc, ema_dec, H_['accumulation_steps'], f"Ep {epoch_num}/{H_['epochs']}"
        )
        
        v_metrics = val_loop(val_dl, ema_enc.model, ema_dec.model, loss_function, curr_kld, device)
        
        logger.info(f"Ep {epoch_num} | T-Loss: {t_metrics[0]:.4f} (MSE:{t_metrics[1]:.2f}) | V-Loss: {v_metrics[0]:.4f} (MSE:{v_metrics[1]:.2f}) | Beta: {curr_kld:.4f}")
        
        if v_metrics[0] < best_val_loss:
            best_val_loss = v_metrics[0]
            torch.save({
                'encoder': ema_enc.model.state_dict(), 
                'decoder': ema_dec.model.state_dict(),
                'config': H_
            }, os.path.join(save_path, "VAE_best_ema.pth"))
            
            raw_enc = encoder.module if hasattr(encoder, 'module') else encoder
            raw_dec = decoder.module if hasattr(decoder, 'module') else decoder
            torch.save({
                'encoder': raw_enc.state_dict(), 
                'decoder': raw_dec.state_dict(),
                'config': H_
            }, os.path.join(save_path, "VAE_best_raw.pth"))
            
            logger.info(f"--> Best Model Saved (Val Loss: {best_val_loss:.4f})")
            
        if epoch_num % 10 == 0:
            torch.save({
                'encoder': ema_enc.model.state_dict(), 
                'decoder': ema_dec.model.state_dict()
            }, os.path.join(save_path, f"VAE_epoch_{epoch_num}.pth"))
