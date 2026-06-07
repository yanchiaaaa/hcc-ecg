import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

from mimic_iv_ecg_dataset import MIMIC_IV_ECG_Dataset
from tqdm import tqdm
import multiprocessing as mp
import gc

def custom_collate_fn(batch):
    """
    Custom collate function for single-sample batches.
    """
    if len(batch) == 1:
        # For batch_size=1, return the sample directly.
        X, label = batch[0]
        if X is None:
            return None, {}
        X = X.unsqueeze(0)  # (L, C) -> (1, L, C)
        return X, label
    else:
        # Batch sizes above 1 are not used in this preprocessing script.
        raise NotImplementedError("Only batch_size=1 is supported.")

@torch.no_grad()
def create_dataset(dataset: Dataset, 
                             target_path: str, 
                             device: str,
                             usage: str):  # Split name.
    try:
        os.makedirs(target_path)
    except:
        pass
    
    BATCH_SIZE = 50000  # Save large training sets in shards.

    data_loader = DataLoader(dataset, 
                            batch_size=1, 
                            num_workers=8, 
                            pin_memory=True,
                            collate_fn=custom_collate_fn)  # Custom collate function.
    
    save_dict = dict()
    exclude_list = []
    anomalistic_data_list = []
    icd_empty_list = []

    part_num = 0
    saved_indices = []  # Original indices retained after filtering.

    print("Starting to create the dataset...")
    for idx, (X, label) in enumerate(tqdm(data_loader)):
        #print(label['subject_id'])
        if X is None:
            icd_empty_list.append(idx)
            continue
        if X.max() > 5 or X.min() < -5:
            anomalistic_data_list.append(idx)
            continue
        if label['hr'] > 99998:
            exclude_list.append(idx)
            continue
        
        # Require a valid age value.
        if 'age' not in label or label['age'] is None or pd.isna(label['age']):
            exclude_list.append(idx)
            continue
         # Move tensors to CPU before saving.
        X_cpu = X.cpu().detach()
        
        # Detach tensor fields inside the label as well.
        label_cpu = {k: (v.cpu().detach() if isinstance(v, torch.Tensor) else v) 
                     for k, v in label.items()}
        save_dict[idx] = {
            'data': X_cpu, 
            'label': label_cpu  # Preserve the original label structure
        }
        saved_indices.append(idx)

        if len(save_dict) >= BATCH_SIZE:
            part_file = os.path.join(
                target_path, 
                f'mimic_vae_train_icd_part{part_num:04d}.pt'
            )
            torch.save(save_dict, part_file)
            print(f"\nSaved shard {part_num}: {len(save_dict):,} samples")
            
            save_dict = dict()
            part_num += 1
            gc.collect()
    
    if usage == 'train':
        if save_dict:
            part_file = os.path.join(
                target_path, 
                f'mimic_vae_train_icd_part{part_num:04d}.pt'
            )
            torch.save(save_dict, part_file)
            print(f"\nSaved final shard: {len(save_dict):,} samples")
            part_num += 1  # Count the final shard.
        metadata = {
            'num_parts': part_num,                      # Number of shards
            'total_samples': len(saved_indices),        # Number of retained samples
            'saved_indices': saved_indices             # Original retained indices
        }
        metadata_file = os.path.join(target_path, f'mimic_vae_train_metadata.pt')
        torch.save(metadata, metadata_file)
        print(f"\nSaved metadata: {metadata_file}")
        # torch.save(save_dict, os.path.join(target_path, 'mimic_vae_train_icd.pt'))
    elif usage == 'test':
        torch.save(save_dict, os.path.join(target_path, 'mimic_vae_test_icd.pt'))
    else:
        torch.save(save_dict, os.path.join(target_path, 'mimic_vae_val_icd.pt'))
    
    print(len(exclude_list))
    print(len(anomalistic_data_list))
    print(len(icd_empty_list))
    
    with open(os.path.join(target_path, f'exclude_list_{usage}.txt'), 'w') as f:
        for idx in exclude_list:
            f.write(str(idx) + '\n')
    with open(os.path.join(target_path, f'anomalistic_data_list_{usage}.txt'), 'w') as f:
        for idx in anomalistic_data_list:
            f.write(str(idx) + '\n')
    with open(os.path.join(target_path, f'icd_empty_list_{usage}.txt'), 'w') as f:
        for idx in icd_empty_list:
            f.write(str(idx) + '\n')

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    device = 'cuda:0'
    target_path = 'data/processed_data_icd'
    model_local_path = 'path/to/Bio_ClinicalBERT'  # Replace with the local Bio_ClinicalBERT path.
    path = 'path/to/mimic-iv-ecg'

    dataset = MIMIC_IV_ECG_Dataset(dataset_path=path,
                                   model_path=model_local_path, # Text encoder path.
                                   device=device,              # Target device.
                                   usage='val',
                                   resample_length=1024,
                                   icd_label=True)

    # # Debug option: use a small subset.
    # # test_dataset = Subset(dataset, range(min(10, len(dataset))))
    # # print(f"Original dataset size: {len(dataset)}")
    # # print(f"Test dataset size: {len(test_dataset)}")
    
    create_dataset(dataset=dataset,   
                            usage='val',
                            target_path=target_path,
                            device=device)
    # dataset = MIMIC_IV_ECG_Dataset(dataset_path=path,
    #                                model_path=model_local_path, # Text encoder path.
    #                                device=device,              # Target device.
    #                                usage='test',
    #                                resample_length=1024,
    #                                icd_label=True)
    # create_dataset(dataset=dataset,   
    #                         usage='test',
    #                         target_path=target_path,
    #                         device=device)
    # dataset = MIMIC_IV_ECG_Dataset(dataset_path=path,
    #                                model_path=model_local_path, # Text encoder path.
    #                                device=device,              # Target device.
    #                                usage='train',
    #                                resample_length=1024,
    #                                icd_label=True)
    # create_dataset(dataset=dataset,   
    #                         usage='train',
    #                         target_path=target_path, 
    #                         device=device)
    torch.cuda.empty_cache()
