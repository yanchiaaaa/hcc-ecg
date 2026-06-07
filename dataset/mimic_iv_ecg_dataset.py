import os
import tqdm
import ast

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy import signal
import wfdb
from wfdb import processing
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import neurokit2 as nk


class MIMIC_IV_ECG_Dataset(Dataset):
    def __init__(self,
                 dataset_path: str, 
                 usage: str='all',  # 'all', 'train', 'val', 'test'
                 num_folds: int=10, 
                 test_fold: int=None, 
                 seed: int=42, 
                 resample_length: int=1024, 
                 icd_label=False,
                 model_path: str = None,
                 device: str = 'cpu'):
        self.model = None
        self.tokenizer = None
        self.icd_label = icd_label

        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.device = device
            self.model.to(self.device)
            self.model.eval() # Set the text encoder to evaluation mode.

        self.resample_length = resample_length
        self.dataset_path = dataset_path

        self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'record_list.csv'), low_memory=False)
        # self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'waveform_note_links.csv'), low_memory=False)

        self.mach_mea = pd.read_csv(os.path.join(self.dataset_path, 'machine_measurements.csv'), low_memory=False)
        self.sheet = pd.merge(self.record_list, self.mach_mea, how='inner', on=['subject_id', 'study_id'])

        #     exclude_list = [eval(x.strip()) for x in f.readlines()]
        # self.sheet.drop(exclude_list, inplace=True)

        # self.demo_label = demo_label
        # self.icd_table = None

        # if demo_label:
        #     icd_table_path = 'data/records_w_diag_icd10_processed.csv'
        #     self.icd_table = pd.read_csv(icd_table_path, index_col='study_id', low_memory=False)
        
        icd_table_path = os.environ.get('HCC_ECG_ICD_TABLE', 'data/records_icdstatement_final.csv')
        icd_table = pd.read_csv(icd_table_path, low_memory=False)
        
        # Merge with fold information based on study_id
        self.sheet = pd.merge(self.sheet, icd_table[['study_id', 'strat_fold','all_diag_all','diag','gender','age']], 
                             how='inner', on='study_id')
        
        # split train, validation and test data based on strat_fold
        if usage in ['train', 'val', 'test']:
            if usage == 'train':
                # First 18 folds (0-17) for training
                sheet_mask = self.sheet['strat_fold'] < 18
            elif usage == 'val':
                # 19th fold (18) for validation
                sheet_mask = self.sheet['strat_fold'] == 18
            elif usage == 'test':
                # 20th fold (19) for testing
                sheet_mask = self.sheet['strat_fold'] == 19
            
            self.sheet = self.sheet[sheet_mask]



    # Preprocessing function for waveform data
    def _waveform_preprocess(self, x: np.ndarray):

        x = np.nan_to_num(x)
        
        # Apply ECG inversion and cleaning for each lead
        sampling_rate = 500  # MIMIC-IV ECG sampling rate
        
        cleaned_leads = []
        for lead_idx in range(x.shape[1]):
            lead = x[:, lead_idx]
            
            # invert_lead, _ = nk.ecg_invert(lead, sampling_rate)
            
            cleaned_lead = nk.ecg_clean(lead, sampling_rate, method="neurokit", powerline=60)
            
            cleaned_leads.append(cleaned_lead)
        
        # Stack cleaned leads back to (L, C) format
        x = np.stack(cleaned_leads, axis=1)
        
        # resample x to intended length
        if self.resample_length:
            x = signal.resample(x, self.resample_length)

        x = torch.as_tensor(x, dtype=torch.float)
        return x

    # Preprocessing function for text label
    def _text_preprocess(self, texts: list):
        # texts: list of 18 reports, where blank is parsed as np.NaN

        text_clean = []
        # wash nan value in texts
        for x in texts:
            if isinstance(x, str):
                text_clean.append(x)

        text_clean = '|'.join(text_clean)

        return text_clean

    def __getitem__(self, idx: int):
        diag = self.sheet.iloc[idx]['diag']
        if isinstance(diag, str):
            try:
                diag_list = ast.literal_eval(diag)
            except (ValueError, SyntaxError):
                diag_list = []
        else:
            diag_list = diag if isinstance(diag, list) else []
        if not diag_list:
            return None, {}
        
        item_path = os.path.join(self.dataset_path, self.sheet['path'].iloc[idx])

        sig, fields = wfdb.rdsamp(item_path)
        x = self._waveform_preprocess(sig)
        
        texts = [self.sheet.iloc[idx][f'report_{x}'] for x in range(18)]
        text = self._text_preprocess(texts)

        rr_interval = self.sheet.iloc[idx]['rr_interval'] / 1000.0

        # abnormal rr interval manually calculate 
        if rr_interval < 0.3 or rr_interval > 1.5:
            heart_rate = None
            for lead in range(12):
                xqrs = processing.XQRS(sig=sig[:, lead], fs=fields['fs'])
                xqrs.detect(verbose=False)
                qrs_inds = xqrs.qrs_inds
                if len(qrs_inds) > 1:
                    rr_intervals = np.diff(qrs_inds) / fields['fs']
                    heart_rate = 60 / np.mean(rr_intervals)
                    break
            # Abort this data in later process
            if heart_rate is None:
                heart_rate = 99999
            #assert heart_rate is not None
                
        else:
            heart_rate = 60.0 / rr_interval

        label_dict = {
                'text': text, 
                'subject_id': self.sheet.iloc[idx]['subject_id'], 
                'study_id': self.sheet.iloc[idx]['study_id'],
                'hr': heart_rate, 
                'age': self.sheet.iloc[idx]['age'],
                'gender': self.sheet.iloc[idx]['gender']
                # 'note_id': self.sheet.iloc[idx]['note_id'], 
                }
        if self.model and self.tokenizer:
            # Text embedding with Bio_ClinicalBERT.
            with torch.no_grad():
                encoded_input = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                model_output = self.model(**encoded_input)
                
                sentence_embeddings = model_output.last_hidden_state[:, 0]  # [CLS] token
                
                text_embed = F.normalize(sentence_embeddings, p=2, dim=1)
                
                text_embed = text_embed.cpu().numpy().squeeze()
                
            label_dict['text_embed'] = text_embed
        
        # Process ICD diagnosis for embedding if icd_label is True
        if self.icd_label and self.model and self.tokenizer:
            label_dict['icd'] = self.sheet.iloc[idx]['all_diag_all']
            label_dict['icd_text'] = self.sheet.iloc[idx]['diag']
            diag_list = self.sheet.iloc[idx]['diag']
            
            # Parse list values stored as strings.
            if isinstance(diag_list, str):
                try:
                    # Safely parse the string with ast.literal_eval.
                    diag_list = ast.literal_eval(diag_list)
                except (ValueError, SyntaxError):
                    # Skip ICD embedding if parsing fails.
                    print("error")
            
            # Method 1: Concatenate all diagnoses with separators
            if isinstance(diag_list, list) and len(diag_list) > 0:
                # Join diagnoses with semicolon separator for clear distinction
                diag_text = "; ".join([str(diag) for diag in diag_list if pd.notna(diag)])
                
                medical_context = f"Medical diagnoses: {diag_text}"
                
                with torch.no_grad():
                    # Encode ICD diagnosis text
                    icd_encoded_input = self.tokenizer(
                        medical_context,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    icd_encoded_input = {k: v.to(self.device) for k, v in icd_encoded_input.items()}
                    
                    # Forward pass
                    icd_model_output = self.model(**icd_encoded_input)
                    
                    # Extract [CLS] token embedding
                    icd_sentence_embeddings = icd_model_output.last_hidden_state[:, 0]
                    
                    # L2 normalization
                    icd_embed = F.normalize(icd_sentence_embeddings, p=2, dim=1)
                    
                    icd_embed = icd_embed.cpu().numpy().squeeze()
                    
                label_dict['icd_embed'] = icd_embed
            else:
                # If no valid diagnoses, assign zero vector
                # label_dict['icd_embed'] = np.zeros(768, dtype=np.float32)
                return None, {}
        # if self.demo_label:
        #     patient_id = label_dict['subject_id']
        #     query = self.patient_table[self.patient_table.index == patient_id] 
        #     label_dict['age'] = query['anchor_age'].to_list()[0]
        #     label_dict['gender'] = query['gender'].to_list()[0]
            
        return x, label_dict

    def __len__(self) -> int:
        return len(self.sheet)
       

class VAE_MIMIC_IV_ECG_Dataset(Dataset):
    def __init__(self, path:str, usage='all'):
        self.path = path
        self.file_list = os.listdir(path)
        # Make sure every time the order of list is the same
        # so as the train and test fold if split in training
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

        if usage == 'test':
            self.file_list = self.file_list[:50000]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index) -> tuple:
        latent_file = self.file_list[index]
        latent_dict = torch.load(os.path.join(self.path, latent_file), map_location='cpu')

        return (latent_dict['data'], latent_dict['label'])

class DictDataset(Dataset):
    def __init__(self, path:str):
        self.data_dict = torch.load(path)
        self.keys = list(self.data_dict.keys()) 

    def __len__(self):
        return len(self.data_dict) 
    
    def __getitem__(self, idx):
        key = self.keys[idx] 
        latent_dict = self.data_dict[key] 

        return latent_dict['data'], latent_dict['label']

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_local_path = 'path/to/Bio_ClinicalBERT'  # Text encoder path.
    path = 'path/to/mimic-iv-ecg'
    
    print("Loading MIMIC_IV_ECG_Dataset...")
    dataset = MIMIC_IV_ECG_Dataset(dataset_path=path,
                                   model_path=model_local_path, # Text encoder path.
                                   device=device,              # Target device.
                                   usage='test',
                                   resample_length=1024,
                                   icd_label=True)  # Enable ICD label embeddings.
    
    print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    print(f"Device: {device}")
    
    print("\n=== Testing first 20 samples ===")
    for i in range(min(20, len(dataset))):
        try:
            x, label = dataset[i]
            print(f"\nSample {i}:")
            #print(label)
            print(f"  ECG shape: {x.shape}")
            print(f"  ECG: {x}")
            print(f"  Subject ID: {label['subject_id']}")
            print(f"  Study ID: {label['study_id']}")
            print(f"  Heart Rate: {label['hr']:.2f}")
            print(f"  Age: {label['age']}")
            print(f"  Gender: {label['gender']}")
            print(f"  Text length: {len(label['text'])} characters")
            print(f"  Text preview: {label['text'][:100]}...")
            print(f"  ICD of Diagnoses: {(label['icd']) if 'icd' in label else 'N/A'}")
            print(f"  ICD Text: {label['icd_text'][:100]}..." if 'icd_text' in label else "  ICD Text: N/A")
            
            if 'text_embed' in label:
                print(f"  Text embedding shape: {label['text_embed'].shape}")
            if 'icd_embed' in label:
                print(f"  ICD embedding shape: {label['icd_embed'].shape}")
                print(f"  Sample diagnoses: {str(dataset.sheet.iloc[i]['diag'])[:200]}...")
        except Exception as e:
            print(f"Error loading sample {i}: {str(e)}")
            break
    
    print(f"\nTest completed!")
    
    # data = MIMIC_IV_ECG_Dataset(dataset_path=path, resample_length=1024, demo_label=True)

    # vae_path = 'mimic_vae.pt'
    # # data = VAE_MIMIC_IV_ECG_Dataset(vae_path, usage='test')
    # data = DictDataset(vae_path)

    # # print(len(data))
    # print(data[397][1])
    # # print(data[397][1])

    # # print(type(data[0][1]['subject_id']))

    # # dataloader = DataLoader(data, 512)
    # # test reading speed
    # for idx, (X, y) in enumerate(tqdm.tqdm(data)):
    #     pass
