

import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
import math
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(epoch, dit_model, ema, optimizer, scheduler,
                    best_val_loss, save_path, is_best=False, logger=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': dit_model.state_dict(),
        'ema_state_dict': ema.state_dict() if ema else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    latest_path = os.path.join(save_path, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    if is_best:
        best_path = os.path.join(save_path, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        if logger:
            logger.info(f"Saved best checkpoint at epoch {epoch} (val_loss={best_val_loss:.4f})")
    if logger:
        logger.info(f"Saved checkpoint at epoch {epoch}")


def load_checkpoint(checkpoint_path, dit_model, ema, optimizer, scheduler,
                    device, logger=None):
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.info(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    if logger:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dit_model.load_state_dict(checkpoint['model_state_dict'])
    if ema and checkpoint.get('ema_state_dict'):
        ema.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    if logger:
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


@torch.no_grad()
def validate_epoch(dataloader, dit_model, diffused_model, device, ema=None):
    
    loss_list = []
    if ema is not None:
        ema.apply_shadow()
    dit_model.eval()

    for data, label in dataloader:
        latent = data.to(device)

        text_embed = label['text_embed'].to(device)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)

        age = label['age'].to(device).squeeze(-1)
        gender = label['gender'].to(device).squeeze(-1)
        hr = label['heart rate'].to(device).squeeze(-1)

        # age_norm = (age - 64.1502) / 17.6533
        # hr_norm = (hr - 81.3950) / 21.4822

        age_norm = (age - 62.6021) / 31.8827
        hr_norm = (hr - 73.9253) / 17.0864

        t = torch.randint(1, diffused_model.config.num_train_timesteps - 1,
                          (latent.shape[0],), device=device)
        noise = torch.randn_like(latent)
        xt = diffused_model.add_noise(latent, noise, t)

        noise_pred = dit_model(
            x=xt, t=t,
            text_embeds=text_embed,
            age=age_norm, gender=gender, hr=hr_norm
        )
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        loss_list.append(loss.item())

    if ema is not None:
        ema.restore()

    local_val_loss = sum(loss_list) / len(loss_list)
    loss_tensor = torch.tensor([local_val_loss], device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def train_model_dit_text_tabular(meta, save_weights_path, dataloader, val_dataloader,
                                  diffused_model, dit_model, h_, logger,
                                  device=None, local_rank=0, is_ddp=False):
    
    if device is None:
        device = torch.device(meta.get('device', 'cuda:0') if torch.cuda.is_available() else "cpu")

    dit_model = dit_model.to(device)

    if is_ddp:
        dit_model = DDP(dit_model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
        if logger: logger.info(f"DDP enabled on rank {local_rank}")

    model_without_ddp = dit_model.module if is_ddp else dit_model
    save_dir = save_weights_path
    exp_type = meta['exp_type']
    exp_num = os.path.basename(save_dir).split('_')[-1]

    resume = h_.get('resume', False)
    checkpoint_latest = os.path.join(save_dir, 'checkpoint_latest.pth')
    resume_from = checkpoint_latest if (resume and os.path.exists(checkpoint_latest)) else None

    if logger:
        mode = "Resume" if resume_from else "New"
        logger.info(f"{mode} | Ablation-D: Text+Tabular | {save_dir} (#{exp_num})")

    optimizer = torch.optim.AdamW(
        dit_model.parameters(), lr=h_['lr'],
        betas=(0.9, 0.999), weight_decay=h_.get('weight_decay', 0.0)
    )

    total_steps = h_['epochs'] * len(dataloader)
    warmup_steps = h_.get('warmup_epochs', 2) * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_ema = h_.get('use_ema', True)
    ema = None
    if use_ema:
        try:
            from utils.ema import EMA
            ema = EMA(model_without_ddp, decay=h_.get('ema_decay', 0.9999),
                      update_after_step=h_.get('ema_update_after', 100),
                      update_every=h_.get('ema_update_every', 10))
            ema = ema.to(device)
            if logger: logger.info("EMA initialized")
        except ImportError:
            ema = None; use_ema = False

    start_epoch, best_val_loss = 0, float('inf')
    if resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, model_without_ddp, ema, optimizer, scheduler, device, logger)

    if logger:
        logger.info(f"Epochs: {start_epoch}→{h_['epochs']} | Device: {device} | DDP: {is_ddp}")

    start_time = time.time()
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, h_['epochs']):
        s_t = time.time()
        loss_list = []
        dit_model.train()

        if is_ddp and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        for data, label in dataloader:
            latent = data.to(device)

            text_embed = label['text_embed'].to(device)
            if text_embed.dim() == 2:
                text_embed = text_embed.unsqueeze(1)

            age = label['age'].to(device).squeeze(-1)
            gender = label['gender'].to(device).squeeze(-1)
            hr = label['heart rate'].to(device).squeeze(-1)

            # age_norm = (age - 64.1502) / 17.6533
            # hr_norm = (hr - 81.3950) / 21.4822

            age_norm = (age - 62.6021) / 31.8827
            hr_norm = (hr - 73.9253) / 17.0864

            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1,
                              (latent.shape[0],), device=device)
            noise = torch.randn_like(latent)
            xt = diffused_model.add_noise(latent, noise, t)

            with torch.cuda.amp.autocast(enabled=True):
                noise_pred = dit_model(
                    x=xt, t=t,
                    text_embeds=text_embed,
                    age=age_norm, gender=gender, hr=hr_norm
                )
                loss = F.mse_loss(noise_pred, noise, reduction='mean')

            loss_list.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), h_.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            if scale_before <= scaler.get_scale():
                scheduler.step()
            if ema is not None and global_step % h_.get('ema_update_every', 10) == 0:
                ema.update()
            global_step += 1

        local_train_loss = sum(loss_list) / len(loss_list)
        train_loss_tensor = torch.tensor([local_train_loss], device=device)
        if is_ddp:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        train_loss = train_loss_tensor.item()

        val_loss = validate_epoch(val_dataloader, model_without_ddp, diffused_model, device, ema)

        if logger:
            ema_status = f"| EMA: {ema.num_updates}" if ema else ""
            logger.info(
                f'Epoch {epoch+1}/{h_["epochs"]} | '
                f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
                f'LR: {scheduler.get_last_lr()[0]:.6f} {ema_status}'
            )

        if not is_ddp or local_rank == 0:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(epoch, model_without_ddp, ema, optimizer, scheduler,
                            best_val_loss, save_dir, is_best, logger)
            if (epoch + 1) % h_.get('save_interval', 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'ema_state_dict': ema.state_dict() if ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        if logger:
            logger.info(f"Time: {time.time()-s_t:.1f}s | Total: {(time.time()-start_time)/60:.1f}min")
            logger.info("-" * 80)

    if (not is_ddp or local_rank == 0) and ema:
        ema.apply_shadow()
        final_path = os.path.join(save_dir, 'dit_text_tabular_ema_final.pth')
        torch.save(model_without_ddp.state_dict(), final_path)
        if logger: logger.info(f"Final EMA weights saved to {final_path}")
        ema.restore()

    if logger:
        logger.info("=" * 80)
        logger.info("Ablation-D (Text+Tabular) training completed.")
