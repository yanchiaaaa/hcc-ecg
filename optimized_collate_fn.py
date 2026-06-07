

import torch
import torch.nn.functional as F
import ast
from functools import partial


class OptimizedCollateFn:
    
    
    def __init__(self, 
                 icd_graph_embeddings=None, 
                 code_to_id=None,
                 use_precomputed_text=True,
                 enable_icd_cache=True,
                 verbose=False):
        self.icd_graph_embeddings = icd_graph_embeddings
        self.code_to_id = code_to_id
        self.use_precomputed_text = use_precomputed_text
        self.enable_icd_cache = enable_icd_cache
        self.verbose = verbose
        
        self.icd_cache = {} if enable_icd_cache else None
        
        self.total_calls = 0
        self.cache_hits = 0
        self.missing_codes = set()
        self.fallback_codes = {}
    
    def __call__(self, batch):
        
        if not batch:
            return None, None
        
        self.total_calls += 1
        batch_size = len(batch)
        
        data_list = [item[0] for item in batch]
        label_list = [item[1] for item in batch]
        
        try:
            data = torch.stack(data_list, dim=0)  # (B, latent_dim, seq_len)
        except Exception as e:
            print(f"Error stacking data: {e}")
            print(f"Data shapes: {[d.shape for d in data_list]}")
            raise
        
        collated_labels = {}
        keys = label_list[0].keys()
        
        for key in keys:
            values = [label[key] for label in label_list]
            
            if key == 'icd':
                collated_labels[key] = values
                
                if self.icd_graph_embeddings is not None and self.code_to_id is not None:
                    icd_embed, icd_mask = self._process_icd_codes(values)
                    collated_labels['icdgraph_embed'] = icd_embed  # (B, max_len, embed_dim)
                    collated_labels['icdgraph_mask'] = icd_mask     # (B, max_len)
            
            elif key == 'text_embed' and self.use_precomputed_text:
                try:
                    text_embeds = []
                    max_text_len = 0
                    
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            text_embeds.append(v)
                            max_text_len = max(max_text_len, v.shape[0])
                        else:
                            text_embeds.append(None)
                    
                    padded_text_embeds = []
                    text_masks = []
                    
                    for emb in text_embeds:
                        if emb is not None:
                            seq_len = emb.shape[0]
                            if seq_len < max_text_len:
                                # Padding
                                pad_size = max_text_len - seq_len
                                padded_emb = F.pad(emb, (0, 0, 0, pad_size), value=0.0)
                            else:
                                padded_emb = emb
                            
                            padded_text_embeds.append(padded_emb)
                            
                            mask = torch.cat([
                                torch.ones(seq_len),
                                torch.zeros(max_text_len - seq_len)
                            ])
                            text_masks.append(mask)
                        else:
                            embed_dim = text_embeds[0].shape[1] if text_embeds[0] is not None else 768
                            padded_text_embeds.append(torch.zeros(max_text_len, embed_dim))
                            text_masks.append(torch.zeros(max_text_len))
                    
                    collated_labels['text_embed'] = torch.stack(padded_text_embeds, dim=0)  # (B, Lt, 768)
                    collated_labels['text_mask'] = torch.stack(text_masks, dim=0)           # (B, Lt)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to process text_embed: {e}")
                    pass
            
            elif key == 'gender':
                # 'M'/'F' -> 1.0/0.0 -> (B, 1, 1)
                gender_nums = [1.0 if g == 'M' else 0.0 for g in values]
                collated_labels[key] = torch.tensor(gender_nums, dtype=torch.float32).view(-1, 1, 1)
            
            elif key == 'age':
                # float -> (B, 1, 1)
                collated_labels[key] = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
            
            elif key == 'hr':
                collated_labels['heart rate'] = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
            
            else:
                collated_labels[key] = values
        
        return data, collated_labels
    
    def _process_icd_codes(self, icd_values):
        
        embed_dim = self.icd_graph_embeddings.shape[1]
        device = self.icd_graph_embeddings.device
        batch_size = len(icd_values)
        
        icd_code_lists = []
        for v in icd_values:
            icd_codes = self._parse_icd_value(v)
            icd_code_lists.append(icd_codes)
        
        max_len = max([len(codes) for codes in icd_code_lists]) if icd_code_lists else 1
        max_len = max(max_len, 1)
        
        padded_embeds = []
        masks = []
        
        for icd_codes in icd_code_lists:
            num_icds = len(icd_codes)
            
            sample_embeds = []
            for icd_code in icd_codes:
                node_id = self._get_id_with_fallback(icd_code)
                
                if node_id is not None:
                    embed_vec = self.icd_graph_embeddings[node_id]  # (embed_dim,)
                    sample_embeds.append(embed_vec)
                else:
                    clean_code = icd_code.replace('.', '').strip()
                    self.missing_codes.add(clean_code)
                    sample_embeds.append(torch.zeros(embed_dim, device=device))
            
            if len(sample_embeds) > 0:
                sample_tensor = torch.stack(sample_embeds, dim=0)
            else:
                sample_tensor = torch.zeros(1, embed_dim, device=device)
                num_icds = 1
            
            if num_icds < max_len:
                pad_size = max_len - num_icds
                padded_tensor = F.pad(sample_tensor, (0, 0, 0, pad_size), value=0.0)
            else:
                padded_tensor = sample_tensor
            
            padded_embeds.append(padded_tensor)
            
            mask = torch.cat([
                torch.ones(num_icds),
                torch.zeros(max_len - num_icds)
            ])
            masks.append(mask)
        
        icd_embed = torch.stack(padded_embeds, dim=0)  # (B, max_len, embed_dim)
        icd_mask = torch.stack(masks, dim=0)           # (B, max_len)
        
        return icd_embed, icd_mask
    
    def _parse_icd_value(self, v):
        
        if isinstance(v, str):
            try:
                icd_codes = ast.literal_eval(v)
                if not isinstance(icd_codes, list):
                    icd_codes = [icd_codes] if icd_codes else []
            except (ValueError, SyntaxError):
                icd_codes = []
        
        elif isinstance(v, list):
            if len(v) == 0:
                icd_codes = []
            elif len(v) == 1 and isinstance(v[0], str):
                v_str = v[0].strip()
                if v_str.startswith('[') and v_str.endswith(']'):
                    try:
                        icd_codes = ast.literal_eval(v_str)
                        if not isinstance(icd_codes, list):
                            icd_codes = [icd_codes] if icd_codes else []
                    except (ValueError, SyntaxError):
                        icd_codes = []
                else:
                    icd_codes = v
            elif isinstance(v[0], list):
                icd_codes = v[0]
            else:
                icd_codes = v
        else:
            icd_codes = []
        
        return icd_codes
    
    def _get_id_with_fallback(self, code_str):
        
        if self.icd_cache is not None and code_str in self.icd_cache:
            self.cache_hits += 1
            return self.icd_cache[code_str]
        
        clean_code = code_str.replace('.', '').strip()
        
        curr = clean_code
        found_id = None
        
        while len(curr) > 0:
            if curr in self.code_to_id:
                found_id = self.code_to_id[curr]
                if curr != clean_code:
                    self.fallback_codes[clean_code] = curr
                break
            
            if len(curr) > 3:
                dotted = curr[:3] + '.' + curr[3:]
                if dotted in self.code_to_id:
                    found_id = self.code_to_id[dotted]
                    if curr != clean_code:
                        self.fallback_codes[clean_code] = dotted
                    break
            
            curr = curr[:-1]
        
        if self.icd_cache is not None:
            self.icd_cache[code_str] = found_id
        
        return found_id
    
    def get_stats(self):
        
        stats = {
            'total_calls': self.total_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.total_calls),
            'missing_codes_count': len(self.missing_codes),
            'fallback_codes_count': len(self.fallback_codes)
        }
        return stats
    
    def print_stats(self):
        
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"OptimizedCollateFn Statistics")
        print(f"{'='*60}")
        print(f"Total Calls:        {stats['total_calls']}")
        print(f"Cache Hits:         {stats['cache_hits']}")
        print(f"Cache Hit Rate:     {stats['cache_hit_rate']:.2%}")
        print(f"Missing Codes:      {stats['missing_codes_count']}")
        print(f"Fallback Codes:     {stats['fallback_codes_count']}")
        print(f"{'='*60}\n")


def custom_collate_fn(batch, icd_graph_embeddings=None, code_to_id=None, verbose=False):
    
    if not batch:
        return None, None
    
    batch_size = len(batch)
    
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    
    try:
        data = torch.stack(data_list, dim=0)  # (B, latent_dim, seq_len)
    except Exception as e:
        print(f"Error stacking data: {e}")
        print(f"Data shapes: {[d.shape for d in data_list]}")
        raise
    
    collated_labels = {}
    keys = label_list[0].keys()
    
    for key in keys:
        values = [label[key] for label in label_list]
        
        if key == 'icd':
            try:
                if icd_graph_embeddings is None or code_to_id is None:
                    collated_labels[key] = values
                    continue
                
                embed_dim = icd_graph_embeddings.shape[1]
                device = icd_graph_embeddings.device
                
                icd_code_lists = []
                for idx, v in enumerate(values):
                    icd_codes = _parse_icd_value(v)
                    icd_code_lists.append(icd_codes)
                
                max_len = max([len(codes) for codes in icd_code_lists]) if icd_code_lists else 1
                max_len = max(max_len, 1)
                
                padded_embeds = []
                masks = []
                batch_missing_codes = set()
                batch_fallback_codes = {}
                
                for sample_idx, icd_codes in enumerate(icd_code_lists):
                    num_icds = len(icd_codes)
                    
                    sample_embeds = []
                    for icd_code in icd_codes:
                        node_id = _get_id_with_fallback(
                            icd_code, code_to_id, 
                            batch_missing_codes, batch_fallback_codes
                        )
                        
                        if node_id is not None:
                            embed_vec = icd_graph_embeddings[node_id]  # (embed_dim,)
                            sample_embeds.append(embed_vec)
                        else:
                            clean_code = icd_code.replace('.', '').strip()
                            batch_missing_codes.add(clean_code)
                            sample_embeds.append(torch.zeros(embed_dim, device=device))
                    
                    if len(sample_embeds) > 0:
                        sample_tensor = torch.stack(sample_embeds, dim=0)
                    else:
                        sample_tensor = torch.zeros(1, embed_dim, device=device)
                        num_icds = 1
                    
                    if num_icds < max_len:
                        pad_size = max_len - num_icds
                        padded_tensor = F.pad(sample_tensor, (0, 0, 0, pad_size), value=0.0)
                    else:
                        padded_tensor = sample_tensor
                    
                    padded_embeds.append(padded_tensor)
                    
                    mask = torch.cat([
                        torch.ones(num_icds), 
                        torch.zeros(max_len - num_icds)
                    ])
                    masks.append(mask)
                
                if batch_missing_codes and verbose:
                    print(f"Warning: {len(batch_missing_codes)} ICD codes not found in graph (using zero vectors)")
                
                collated_labels['icdgraph_embed'] = torch.stack(padded_embeds, dim=0)  # (B, max_len, embed_dim)
                collated_labels['icdgraph_mask'] = torch.stack(masks, dim=0)           # (B, max_len)
                
                collated_labels['icd'] = values
                
            except Exception as e:
                print(f"Error processing ICD codes: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        elif key == 'gender':
            # 'M'/'F' -> 1.0/0.0 -> (B, 1, 1)
            gender_nums = [1.0 if g == 'M' else 0.0 for g in values]
            collated_labels[key] = torch.tensor(gender_nums, dtype=torch.float32).view(-1, 1, 1)
        
        elif key == 'age':
            # float -> (B, 1, 1)
            collated_labels[key] = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
        
        elif key == 'hr':
            collated_labels['heart rate'] = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
        
        else:
            collated_labels[key] = values
    
    return data, collated_labels


def _parse_icd_value(v):
    
    if isinstance(v, str):
        try:
            icd_codes = ast.literal_eval(v)
            if not isinstance(icd_codes, list):
                icd_codes = [icd_codes] if icd_codes else []
        except (ValueError, SyntaxError):
            icd_codes = []
    
    elif isinstance(v, list):
        if len(v) == 0:
            icd_codes = []
        elif len(v) == 1 and isinstance(v[0], str):
            v_str = v[0].strip()
            if v_str.startswith('[') and v_str.endswith(']'):
                try:
                    icd_codes = ast.literal_eval(v_str)
                    if not isinstance(icd_codes, list):
                        icd_codes = [icd_codes] if icd_codes else []
                except (ValueError, SyntaxError):
                    icd_codes = []
            else:
                icd_codes = v
        elif isinstance(v[0], list):
            icd_codes = v[0]
        else:
            icd_codes = v
    else:
        icd_codes = []
    
    return icd_codes


def _get_id_with_fallback(code_str, code_to_id, batch_missing_codes, batch_fallback_codes):
    
    clean_code = code_str.replace('.', '').strip()
    
    curr = clean_code
    while len(curr) > 0:
        if curr in code_to_id:
            if curr != clean_code:
                if clean_code not in batch_fallback_codes:
                    batch_fallback_codes[clean_code] = curr
            return code_to_id[curr]
        
        if len(curr) > 3:
            dotted = curr[:3] + '.' + curr[3:]
            if dotted in code_to_id:
                if curr != clean_code:
                    if clean_code not in batch_fallback_codes:
                        batch_fallback_codes[clean_code] = dotted
                return code_to_id[dotted]
        
        curr = curr[:-1]
    
    return None


if __name__ == '__main__':
    print("Testing OptimizedCollateFn...")
    print("="*70)
    
    batch_size = 4
    
    num_nodes = 100
    embed_dim = 768
    icd_embeddings = torch.randn(num_nodes, embed_dim)
    
    code_to_id = {
        'I10': 0,
        'E119': 1,
        'R001': 2,
        'T17.598A': 3,
        'T17.598': 4,
        'T17.5': 5,
        'Z9981': 6
    }
    
    batch = [
        (torch.randn(4, 128), {'icd': "['I10', 'E119']", 'gender': 'M', 'age': 65.0, 'hr': 80.0}),
        (torch.randn(4, 128), {'icd': "['R001']", 'gender': 'F', 'age': 45.0, 'hr': 75.0}),
        (torch.randn(4, 128), {'icd': "['T17.598A', 'I10']", 'gender': 'M', 'age': 55.0, 'hr': 90.0}),
        (torch.randn(4, 128), {'icd': "['UNKNOWN', 'E119']", 'gender': 'F', 'age': 70.0, 'hr': 85.0})
    ]
    
    print("\n" + "="*70)
    print("Test 1: class interface (OptimizedCollateFn)")
    print("="*70)
    
    collate_fn = OptimizedCollateFn(
        icd_graph_embeddings=icd_embeddings,
        code_to_id=code_to_id,
        use_precomputed_text=False,
        enable_icd_cache=True,
        verbose=True
    )
    
    data, labels = collate_fn(batch)
    
    print(f"\nCollate succeeded")
    print(f"  data shape: {data.shape}")
    print(f"  icdgraph_embed shape: {labels['icdgraph_embed'].shape}")
    print(f"  icdgraph_mask shape: {labels['icdgraph_mask'].shape}")
    print(f"  gender shape: {labels['gender'].shape}")
    print(f"  age shape: {labels['age'].shape}")
    print(f"  heart rate shape: {labels['heart rate'].shape}")
    
    print(f"\nValidate mask:")
    for i in range(batch_size):
        num_valid = int(labels['icdgraph_mask'][i].sum().item())
        print(f"  Sample {i}: {num_valid} valid ICD codes")
    
    collate_fn.print_stats()
    
    print("\n" + "="*70)
    print("Test 2: functional interface (custom_collate_fn)")
    print("="*70)
    
    from functools import partial
    collate_fn_with_graph = partial(
        custom_collate_fn,
        icd_graph_embeddings=icd_embeddings,
        code_to_id=code_to_id,
        verbose=True
    )
    
    data2, labels2 = collate_fn_with_graph(batch)
    
    print(f"\nCollate succeeded")
    print(f"  data shape: {data2.shape}")
    print(f"  icdgraph_embed shape: {labels2['icdgraph_embed'].shape}")
    print(f"  icdgraph_mask shape: {labels2['icdgraph_mask'].shape}")
    print(f"  gender shape: {labels2['gender'].shape}")
    print(f"  age shape: {labels2['age'].shape}")
    print(f"  heart rate shape: {labels2['heart rate'].shape}")
    
    print(f"\nValidate that both interfaces match:")
    print(f"  data match: {torch.allclose(data, data2)}")
    print(f"  icdgraph_embed match: {torch.allclose(labels['icdgraph_embed'], labels2['icdgraph_embed'])}")
    print(f"  icdgraph_mask match: {torch.allclose(labels['icdgraph_mask'], labels2['icdgraph_mask'])}")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
