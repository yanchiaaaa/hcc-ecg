#     cd /path/to/HCC-ECG
#     python HCCECG.py config/PTB.json
#     cd /path/to/HCC-ECG
#     torchrun --nproc_per_node=N HCCECG.py config/PTB.json
#     CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 HCCECG.py config/PTB.json
#     torchrun --nproc_per_node=2 --master_port=29501 HCCECG.py config/PTB.json

import argparse 
import json
import logging
import os
import torch 
import traceback
from torch.utils.data import DataLoader 
from diffusers import DDPMScheduler
from dataset.mimic_iv_ecg_dataset import DictDataset
from optimized_collate_fn import OptimizedCollateFn
from icd_graph_loader import ICDGraphEmbeddingLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def parse_arg():
    parser = argparse.ArgumentParser(description='HCC-ECG Training') 
    parser.add_argument('config', help='Root of training configuration')

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    with open(args.config, 'r') as f:
        config = json.load(f)

    meta = config['meta']
    roots = config['dependencies']
    h_ = config['hyper_para']

    if "LOCAL_RANK" in os.environ:
        is_ddp = True
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_master = (local_rank == 0)
    else:
        is_ddp = False
        local_rank = 0
        is_master = True
        dev = meta.get('device')
        if dev:
            device = torch.device(dev)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(f"meta.device={dev} requires CUDA, but torch.cuda.is_available() is False")
            idx = device.index if device.index is not None else 0
            torch.cuda.set_device(idx)
            device = torch.device(f'cuda:{idx}')

    resume = h_.get('resume', False)
    resume_checkpoint_dir = h_.get('resume_checkpoint_dir', None)
    
    if is_master:
        if resume and resume_checkpoint_dir:
            save_weights_path = resume_checkpoint_dir
            exp_num = os.path.basename(save_weights_path).split('_')[-1]
            logger_name = os.path.basename(save_weights_path)
        else:
            k_max = 0
            for item in os.listdir(roots['checkpoints_dir']):
                if meta['exp_type'] + "_" in item:
                    try:
                        k = int(item.split('_')[-1]) 
                        k_max = k if k > k_max else k_max
                    except ValueError:
                        continue
            exp_num = k_max + 1
            save_weights_path = os.path.join(roots['checkpoints_dir'], f"{meta['exp_type']}_{exp_num}")
            logger_name = f"{meta['exp_type']}_{exp_num}"
            
            try:
                os.makedirs(save_weights_path, exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to create directory {save_weights_path}: {e}")
    
    if is_ddp:
        if is_master:
            save_weights_path_list = [save_weights_path]
        else:
            save_weights_path_list = [None]
        
        dist.broadcast_object_list(save_weights_path_list, src=0)
        save_weights_path = save_weights_path_list[0]
        logger_name = os.path.basename(save_weights_path)
        
        dist.barrier()

    if is_master:
        logger = logging.getLogger(logger_name)
        logger.setLevel('INFO')
        fh = logging.FileHandler(os.path.join(save_weights_path, 'train.log'), encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f"🚀 DDP Training Started: Rank 0/{dist.get_world_size() if is_ddp else 1}")
        logger.info(meta)
        logger.info(h_)
    else:
        logger = logging.getLogger(f"{logger_name}_rank{local_rank}")
        logger.setLevel('ERROR')
        logger.addHandler(logging.NullHandler())

    icd_loader = ICDGraphEmbeddingLoader(
        graph_data_path=roots.get('icd_graph_path', 'path/to/icd_graph_data.pt'),
        embeddings_path=roots.get('icd_embeddings_path', 'path/to/icd_hyperbolic_best.pth'),
        special_tokens=['NORM'],
        logger=logger if is_master else None,
    )
    icd_embeddings, code_to_id = icd_loader.load()

    train_dataset = DictDataset(roots['dataset_path'])

    collate_fn_handler = OptimizedCollateFn(
    icd_graph_embeddings=icd_embeddings,
    code_to_id=code_to_id,
    use_precomputed_text=True,
    enable_icd_cache=True,
    verbose=False
    )


    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle_option = False 
    else:
        train_sampler = None
        shuffle_option = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=h_['batch_size'],
        shuffle=shuffle_option,
        sampler=train_sampler,
        collate_fn=collate_fn_handler,
        drop_last=True
    )

    
    val_dataset = DictDataset(roots['val_dataset_path'])

    if is_ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None
        
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=h_['batch_size'], 
        shuffle=False, 
        sampler=val_sampler,
        collate_fn=collate_fn_handler
    )

    if is_master:
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    vae_latent_dim = h_.get('vae_latent_dim', 4)
    n_channels = vae_latent_dim if meta.get('vae_latent', True) else 12
    ablation_mode = h_.get('ablation_mode', None)  # None | "icd_only" | "text_only" | "tabular_only" | "text_tabular" | "icd_tabular"

    common_kwargs = dict(
        in_channels=n_channels,
        seq_length=128,
        hidden_size=h_.get('dit_hidden_size', 512),
        depth=h_.get('dit_depth', 12),
        num_heads=h_.get('dit_num_heads', 8),
        mlp_ratio=h_.get('mlp_ratio', 4.0),
        dropout=h_.get('dit_dropout', 0.0),
    )

    if ablation_mode not in (None, 'text_tabular', 'icd_tabular'):
        raise ValueError("This open-source package includes only the full model, text_tabular, and icd_tabular ablations.")

    if ablation_mode == 'text_tabular':
        from module.dit_ablation_text_tabular import DiT_TextTabular_ECG
        model = DiT_TextTabular_ECG(**common_kwargs, text_embed_dim=h_.get('text_embed_dim', 768))
        if is_master: logger.info("[Ablation-D] DiT Text+Tabular (No ICD) model created")

    elif ablation_mode == 'icd_tabular':
        from module.dit_ablation_icd_tabular import DiT_ICDTabular_ECG
        model = DiT_ICDTabular_ECG(**common_kwargs, icd_embed_dim=h_.get('icd_embed_dim', 768))
        if is_master: logger.info("[Ablation-E] DiT ICD+Tabular (No Text) model created")

    else :
        from module.dit_tri_stream_noproj_newcfg import DiT_TripleStream_ECG
        model = DiT_TripleStream_ECG(
            **common_kwargs,
            icd_embed_dim=h_.get('icd_embed_dim', 768),
            text_embed_dim=h_.get('text_embed_dim', 768),
            use_rope=h_.get('use_rope', False)
        )
        if is_master: logger.info("DiT Tri-Stream (ICD + Text + Tabular) model created")


    if is_master:
        logger.info(f"  Hidden={common_kwargs['hidden_size']}, Depth={common_kwargs['depth']}, "
                     f"Heads={common_kwargs['num_heads']}, Dropout={common_kwargs['dropout']}")

    diffused_model = DDPMScheduler(
        num_train_timesteps=h_['num_train_steps'], 
        beta_start=h_['beta_start'], 
        beta_end=h_['beta_end']
    )

    train_kwargs = dict(
        meta=meta, save_weights_path=save_weights_path,
        dataloader=train_dataloader, val_dataloader=val_dataloader,
        diffused_model=diffused_model, dit_model=model,
        h_=h_, logger=logger, device=device,
        local_rank=local_rank, is_ddp=is_ddp
    )

    if ablation_mode == 'text_tabular':
        from utils.train_ablation_text_tabular import train_model_dit_text_tabular
        train_model_dit_text_tabular(**train_kwargs)

    elif ablation_mode == 'icd_tabular':
        from utils.train_ablation_icd_tabular import train_model_dit_icd_tabular
        train_model_dit_icd_tabular(**train_kwargs)

    else :
        from utils.train_dit_tri_stream import train_model_dit_dual
        train_model_dit_dual(**train_kwargs)


if __name__ == '__main__': 
    main()
