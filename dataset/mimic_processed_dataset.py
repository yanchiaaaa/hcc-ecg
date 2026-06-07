import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MIMIC_IV_ECG_Processed_Dataset(Dataset):
    """
    Dataset wrapper for preprocessed MIMIC-IV ECG files.
    
    Supports two storage formats:
    1. Sharded training files with metadata.
    2. Single .pt files for validation and test splits.
    """
    
    def __init__(self, data_path: str, usage: str = 'train', preload_shards: bool = False):
        """
        Args:
            data_path: Directory containing the processed files.
            usage: 'train', 'test', or 'val'.
            preload_shards: Whether to preload all shards into memory.
        """
        self.data_path = data_path
        self.usage = usage
        self.preload_shards = preload_shards
        
        if usage == 'train':
            self._load_sharded_data()
        else:
            self._load_single_file()
    
    def _load_sharded_data(self):
        """Load sharded training data."""
        metadata_file = os.path.join(self.data_path, 'mimic_vae_train_metadata.pt')
        metadata = torch.load(metadata_file, map_location='cpu')
        
        self.num_parts = metadata['num_parts']
        self.total_samples = metadata['total_samples']
        self.saved_indices = metadata['saved_indices']  # Original indices retained after filtering.
        
        self.part_files = []
        for part_num in range(self.num_parts):
            part_file = os.path.join(
                self.data_path, 
                f'mimic_vae_train_icd_part{part_num:04d}.pt'
            )
            self.part_files.append(part_file)
        
        # Optionally preload all shards into memory.
        if self.preload_shards:
            print("Preloading all shards into memory...")
            self.all_data = {}
            for part_file in self.part_files:
                if os.path.exists(part_file):
                    part_data = torch.load(part_file, map_location='cpu')
                    self.all_data.update(part_data)
                    print(f"Loaded: {os.path.basename(part_file)}")
            print(f"Preloading complete, total samples: {len(self.all_data)}")
        else:
            # Cache shards loaded on demand.
            self._part_cache = {}
    
    def _load_single_file(self):
        """Load a single validation/test file."""
        file_path = os.path.join(self.data_path, f'mimic_vae_{self.usage}_icd.pt')
        self.data_dict = torch.load(file_path, map_location='cpu')
        self.keys = list(self.data_dict.keys())
        self.total_samples = len(self.keys)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if self.usage == 'train':
            return self._get_sharded_item(idx)
        else:
            return self._get_single_item(idx)
    
    def _get_sharded_item(self, idx):
        """Return one item from the sharded data."""
        # Map to the original sample index.
        original_idx = self.saved_indices[idx]
        
        if self.preload_shards:
            if original_idx in self.all_data:
                item = self.all_data[original_idx]
                return item['data'], item['label']
            else:
                return None, {}
        else:
            # Find the shard containing this sample.
            # A shard index map could be added; linear lookup is kept for simplicity.
            for part_file in self.part_files:
                if part_file not in self._part_cache:
                    if os.path.exists(part_file):
                        self._part_cache[part_file] = torch.load(part_file, map_location='cpu')
                
                part_data = self._part_cache[part_file]
                if original_idx in part_data:
                    item = part_data[original_idx]
                    return item['data'], item['label']
            
            return None, {}
    
    def _get_single_item(self, idx):
        """Return one item from a single-file split."""
        key = self.keys[idx]
        item = self.data_dict[key]
        return item['data'], item['label']
    
    def clear_cache(self):
        """Clear the shard cache and release memory."""
        if hasattr(self, '_part_cache'):
            self._part_cache.clear()
        import gc
        gc.collect()

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATA_PATH = 'data/processed_data_icd'
    
    test_dataset = MIMIC_IV_ECG_Processed_Dataset(
        data_path=DATA_PATH,
        usage='test'
    )
    
    print(f"\nMIMIC test dataset loaded, total samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False, # Keep deterministic ordering for inspection.
        num_workers=0 
    )
    
    print("\n" + "="*50)
    print("Detailed MIMIC item structure")
    print("="*50)

    for idx, (batch_data, batch_labels) in enumerate(test_loader):
        if idx >= 2:
            break
            
        print("\n" + "="*50)
        print(f"      Sample {idx + 1} structure")
        print("="*50)

        print(f"[Input X]")
        print(f"  -> Shape: {batch_data.shape}")
        print(f"  -> Dtype: {batch_data.dtype}")
        
        print(f"\n[Label fields]")
        
        keys_to_print = ['text', 'subject_id', 'study_id', 'hr', 'age', 'gender', 'icd', 'icd_text', 'text_embed', 'icd_embed']
        
        for key in keys_to_print:
            if key not in batch_labels:
                if key == 'icd_text' and 'diag' in batch_labels:
                    value = batch_labels['diag']
                else:
                    continue
            else:
                value = batch_labels[key]
            
            if isinstance(value, torch.Tensor):
                print(f"  -> [Tensor]  {key:12} : shape = {list(value.shape)}")
                
            elif isinstance(value, list):
                content = value[0] if len(value) > 0 else ""
                
                if isinstance(content, str):
                    print(f"  -> [List]    {key:12} : length = {len(content)} (example: {content[:80]}...)" if len(content)>80 else f"  -> [List]    {key:12} : length = {len(content)} (example: {content})")
                
                elif isinstance(content, (list, tuple)):
                    print(f"  -> [List]    {key:12} : length = {len(value)} (example: {content[0] if len(content)>0 else []})")
                
                else:
                    print(f"  -> [List]    {key:12} : length = {len(value)} (example: {content})")
            
            elif isinstance(value, str):
                print(f"  -> [String]  {key:12} : length = {len(value)} (example: {value[:80]}...)" if len(value)>80 else f"  -> [String]  {key:12} : length = {len(value)} (example: {value})")
                
            else:
                print(f"  -> [Other]   {key:12} : type = {type(value).__name__}, value = {value}")

        print("\n" + "-"*50)
        print(f"Sample {idx + 1} full preview")
        print("-"*50)
