import os
import sys

import sys
from dataset.mimic_processed_dataset import MIMIC_IV_ECG_Processed_Dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import multiprocessing as mp
import sys
from VAE.vae_model import VAE_Encoder



@torch.no_grad()
def encode_dataset_to_latent(dataset: Dataset, 
                             vae_encoder: VAE_Encoder, 
                             target_path: str, 
                             device: str,
                             usage: str):
    """
    使用 VAE 编码器将数据集编码到潜在空间
    
    参数:
        dataset: ECG 数据集
        vae_encoder: VAE 编码器模型
        target_path: 保存路径
        device: 设备 ('cuda:0', 'cuda:1', 'cpu')
        usage: 数据集类型 ('train', 'val', 'test')
    """
    try:
        os.makedirs(target_path, exist_ok=True)
    except:
        pass

    vae_encoder.to(device)
    vae_encoder.eval()  # 设置为评估模式
    save_dict = dict()

    data_loader = DataLoader(dataset, 
                            batch_size=1, 
                            num_workers=0, 
                            pin_memory=False) 

    print(f"Starting to encode {usage} dataset with VAE Improved...")
    for idx, (X, label) in enumerate(tqdm(data_loader, desc=f"Encoding {usage}")):
        X = X.to(device)
        
        # 🔥 修正数据形状：保留 batch 维度，只压缩中间的维度
        # 从 (B, L, 1, C) 或 (B, 1, L, C) -> (B, L, C)
        if X.dim() == 4:
            # 找到大小为1的维度（排除 batch 维度）
            # 通常是 (1, 1024, 1, 12) 或 (1, 1, 1024, 12)
            if X.shape[2] == 1:
                X = X.squeeze(2)  # (B, L, 1, C) -> (B, L, C)
            elif X.shape[1] == 1:
                X = X.squeeze(1)  # (B, 1, L, C) -> (B, L, C)
        
        # 调试：检查第一个 batch 的形状
        if idx == 0:
            print(f"输入形状: {X.shape}")
        
        # 确保是 3D: (B, L, C)
        if X.dim() != 3:
            raise ValueError(f"Expected 3D input (B, L, C), got shape: {X.shape}")

        # X: (B, L, C) -> latent: (B, latent_dim, latent_length)
        # 对于 latent_dim=8, layers=3: (B, 8, 128)
        latent, _, __ = vae_encoder(X)
        latent = latent.squeeze(0)  # (B, latent_dim, latent_length) -> (latent_dim, latent_length)

        # 直接保存 latent 和 label
        save_dict[idx] = {
            'data': latent.cpu(), 
            'label': label
        }
    
    # 根据 usage 保存最终文件（使用 pssm_vae 前缀）
    if usage == 'train':
        save_file = os.path.join(target_path, 'hcc_vae_train.pt')
    elif usage == 'test':
        save_file = os.path.join(target_path, 'hcc_vae_test.pt')
    else:
        save_file = os.path.join(target_path, 'hcc_vae_val.pt')
    
    torch.save(save_dict, save_file)
    print(f"✓ Encoded {len(save_dict)} samples for {usage} dataset.")
    print(f"✓ Saved to: {save_file}")

if __name__ == '__main__':
    # 设置多进程启动方式
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # ==================== 配置 ====================
    device = 'cuda:0'
    data_path = ''  # 预处理数据路径
    target_path = ''  # 保存路径
    
    vae_checkpoint = ''
    
    print("="*60)
    print("VAE (Improved) 潜在向量编码")
    print("="*60)
    print(f"设备: {device}")
    print(f"数据路径: {data_path}")
    print(f"保存路径: {target_path}")
    print(f"模型路径: {vae_checkpoint}")
    print("="*60)
    
    # 加载 VAE 编码器
    encoder = VAE_Encoder()
    
    # 加载权重
    checkpoint = torch.load(vae_checkpoint, map_location=device)
    # 你的保存代码 key 是 'encoder'，这里直接加载即可
    encoder.load_state_dict(checkpoint['encoder'])
    
    print(f"✓ Model loaded successfully!")
    # 打印一下参数量确认没加载错
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    
    # ==================== 编码验证集 ====================
    print("\n" + "="*60)
    print("编码验证集...")
    print("="*60)
    val_dataset = MIMIC_IV_ECG_Processed_Dataset(
        data_path=data_path, 
        usage='val',
        preload_shards=True
    )
    encode_dataset_to_latent(
        dataset=val_dataset,   
        vae_encoder=encoder,
        target_path=target_path,
        device=device,
        usage='val'
    )
    torch.cuda.empty_cache()
    
    # ==================== 编码测试集 ====================
    print("\n" + "="*60)
    print("编码测试集...")
    print("="*60)
    test_dataset = MIMIC_IV_ECG_Processed_Dataset(
        data_path=data_path, 
        usage='test',
        preload_shards=True
    )
    encode_dataset_to_latent(
        dataset=test_dataset,   
        vae_encoder=encoder,
        target_path=target_path,
        device=device,
        usage='test'
    )
    torch.cuda.empty_cache()
    
    # ==================== 编码训练集 ====================
    print("\n" + "="*60)
    print("编码训练集...")
    print("="*60)
    train_dataset = MIMIC_IV_ECG_Processed_Dataset(
        data_path=data_path, 
        usage='train',
        preload_shards=True
    )
    encode_dataset_to_latent(
        dataset=train_dataset,   
        vae_encoder=encoder, 
        target_path=target_path, 
        device=device,
        usage='train'
    )
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✓ 所有数据集编码完成！")
    print("="*60)
    print(f"生成的文件:")
    print(f"  - {target_path}/hcc_vae_train.pt")
    print(f"  - {target_path}/hcc_vae_val.pt")
    print(f"  - {target_path}/hcc_vae_test.pt")
    print("="*60)