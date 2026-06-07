#!/usr/bin/env python3
"""
Preprocess records_w_diag_icd10.csv.
Convert ICD-10 codes into diagnosis descriptions.
"""

import pandas as pd
import ast
import gzip
from pathlib import Path

RECORDS_FILE = "data/records_w_diag_icd10.csv"
ICD_DICT_FILE = "data/d_icd_diagnoses.csv.gz"
TEXT_FILE = "data/machine_measurements.csv"
OUTPUT_FILE = "data/records_icdstatement_final.csv"

COLUMNS_TO_KEEP = [
    "file_name",      # ECG file index.
    "study_id",       # Study identifier.
    "subject_id",     # Patient identifier for leakage control.
    "all_diag_all",   # ICD-10 diagnosis sequence.
    "gender",         # Demographic condition.
    "age",            # Demographic condition.
    "strat_fold",     # Stratified split assignment.
    "fold"            # Legacy random split assignment.
]


def load_icd_dictionary():
    """
    Load the ICD-10 diagnosis dictionary.
    Returns: dict {icd_code: long_title}.
    """
    print("Load the ICD-10 diagnosis dictionary....")
    
    with gzip.open(ICD_DICT_FILE, 'rt') as f:
        df_icd = pd.read_csv(f)
    
    df_icd10 = df_icd[df_icd['icd_version'] == 10].copy()
    
    icd_dict = dict(zip(df_icd10['icd_code'], df_icd10['long_title']))
    
    print(f"Loaded {len(icd_dict)} ICD-10 diagnosis codes")
    
    return icd_dict


def convert_icd_codes_to_concepts(icd_codes_str, icd_dict):
    """
    Convert an ICD-code list string to diagnosis descriptions.
    
    Args:
        icd_codes_str: List encoded as a string, e.g. "['K7469', 'E871', 'R64']"
        icd_dict: Mapping from ICD code to diagnosis description.
    
    Returns:
        Diagnosis description list, e.g. ['concept1', 'concept2', 'concept3'].
    """
    if pd.isna(icd_codes_str) or icd_codes_str == '[]':
        return []
    
    try:
        icd_codes = ast.literal_eval(icd_codes_str)
        
        concepts = []
        for code in icd_codes:
            if code in icd_dict:
                concepts.append(icd_dict[code])
            else:
                # Mark codes missing from the dictionary.
                concepts.append(f"Unknown({code})")
                # Omit noisy per-code warnings.
        
        return concepts
    
    except Exception as e:
        print(f"Error parsing ICD codes: {icd_codes_str}, error: {e}")
        return []

def is_clean_cardiac_code(code):
    """
    Rule-based ICD filtering by ECG morphology relevance.
    """
    c = str(code).replace('.', '').strip().upper()
    
    
    # Chapter E: electrolyte disorders and thyrotoxicosis.
    if c.startswith('E87'): return True  # K+, Na+, Ca2+ imbalance
    if c.startswith('E05'): return True  # Thyrotoxicosis
    
    # Chapter T: cardiotoxic medications.
    # T43.01: TCA (Long QT)
    if c.startswith('T4301'): return True
    # T46: Digoxin, Antiarrhythmics, CCB
    if c.startswith('T46'): return True
    
    # Chapter Q: congenital heart disease.
    if c.startswith(('Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28')): 
        return True
        
    # Chapter J: pulmonary disease affecting voltage or axis.
    if c.startswith('J44'): return True
    
    # Chapter I: circulatory-system diagnoses.
    if c.startswith('I'):
        # --- Chapter I blacklist ---
        # Hypertension history only.
        if c == 'I10' or c.startswith(('I11', 'I12', 'I13', 'I15')): return False
        # Chronic/anatomic ischemia without infarction evidence.
        if c.startswith(('I251', 'I258', 'I259')): return False
        # Heart failure syndrome, mostly echocardiographic.
        if c.startswith('I50'): return False
        # Non-cardiac vascular disease.
        if c.startswith(('I6', 'I7', 'I8', 'I9')): return False
        
        # --- Keep remaining Chapter I codes ---
        return True

    return False

def check_text_is_normal(text_list):
    """
    Use report text to detect normal or near-normal ECG cases.
    """
    if not text_list: return False
    
    full_text = ""
    if isinstance(text_list, list):
        full_text = " ".join([str(t) for t in text_list]).lower()
    else:
        full_text = str(text_list).lower()
        
    normal_keywords = [
        'normal ecg', 
        'normal sinus rhythm', 
        'within normal limits',
        'no significant abnormality'
    ]
    
    abnormal_keywords = [
        'abnormal ecg', 'infarct', 'ischemia', 'block', 'fibrillation', 
        'flutter', 'hypertrophy', 'injury', 'st elevation', 'st depression'
    ]
    
    for kw in abnormal_keywords:
        if kw in full_text:
            return False
            
    for kw in normal_keywords:
        if kw in full_text:
            return True
            
    if 'sinus rhythm' in full_text and len(full_text) < 15: 
        return True
        
    return False

def preprocess_records():
    """
    Main preprocessing routine.
    """
    print("=" * 60)
    print("Preprocessing records_w_diag_icd10.csv")
    print("=" * 60)
    
    icd_dict = load_icd_dictionary()
    
    print("\nReading source records...")
    df_records = pd.read_csv(RECORDS_FILE)
    
    print("Reading ECG report text...")
    try:
        actual_text_file = TEXT_FILE
        df_text = pd.read_csv(actual_text_file)
        text_cols_to_keep = ['study_id'] + [f'report_{i}' for i in range(18) if f'report_{i}' in df_text.columns]
        df_text = df_text[text_cols_to_keep]
        
        print("Merging diagnoses with report text...")
        df = pd.merge(df_records, df_text, on='study_id', how='left')
    except Exception as e:
        print(f"Failed to merge report text. Error: {e}")
        df = df_records

    if 'report_0' not in df.columns:
        print("Warning: report columns are missing; text-based validation is disabled.")
    
    print(f"Merged data shape: {df.shape}")
    print(f"Merged columns: {df.columns.tolist()}")
    
    print("\nSelecting required columns...")
    report_columns = [f'report_{x}' for x in range(18) if f'report_{x}' in df.columns]
    df_selected = df[COLUMNS_TO_KEEP + report_columns].copy()

    print("\nFiltering ICD codes and converting descriptions...")
    valid_rows = []
    
    for idx, row in df_selected.iterrows():
        icd_codes_str = row['all_diag_all']
        if pd.isna(icd_codes_str) or icd_codes_str == '[]':
            continue  # Skip rows without ICD codes.
        
        try:
            icd_codes = ast.literal_eval(icd_codes_str)
        except:
            continue
            
        if not icd_codes:
            continue
            
        cleaned_codes = [code for code in icd_codes if is_clean_cardiac_code(code)]
        
        # Fall back to text-based normal-label recovery.
        if len(cleaned_codes) > 0:
            cleaned_codes_str = str(cleaned_codes)
            concepts = convert_icd_codes_to_concepts(str(cleaned_codes), icd_dict)
            concepts_str = str(concepts)
        else:
            texts = [row[col] for col in report_columns]
            valid_texts = [x for x in texts if isinstance(x, str)]
            text_joined = '|'.join(valid_texts)
            
            if check_text_is_normal(text_joined):
                cleaned_codes_str = "['NORM']"
                concepts_str = "['Normal ECG']"
            else:
                continue # Drop rows without retained ICD codes or a valid normal-text fallback.
            
        new_row = {k: row[k] for k in COLUMNS_TO_KEEP}
        new_row['all_diag_all'] = cleaned_codes_str   # Store filtered ICD codes.
        new_row['diag'] = concepts_str                # Store diagnosis descriptions.
        valid_rows.append(new_row)
        
    df_final = pd.DataFrame(valid_rows)

    column_order = [
        "file_name",
        "study_id",
        "subject_id",
        "all_diag_all",
        "diag",  # New column.
        "gender",
        "age",
        "strat_fold",
        "fold"
    ]
    df_final = df_final[column_order]
    
    print(f"\nSaving processed data to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)
    print(f"Records before filtering: {len(df)}")
    print(f"Records after filtering: {len(df_final)}")
    print(f"\nPreview of the first 3 rows:")
    print("-" * 60)
    
    for idx in range(min(3, len(df_final))):
        row = df_final.iloc[idx]
        print(f"\nRecord {idx + 1}:")
        print(f"  file_name: {row['file_name']}")
        print(f"  study_id: {row['study_id']}")
        print(f"  subject_id: {row['subject_id']}")
        print(f"  gender: {row['gender']}, age: {row['age']}")
        print(f"  all_diag_all: {row['all_diag_all']}")
        print(f"  diag: {row['diag']}")
    
    print("\n" + "=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_records()
