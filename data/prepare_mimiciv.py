import logging
import random
import pickle
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re

from utils import (
    preprocess_documents,
    reformat_icd,
    reformat_code_dataframe,
    filter_codes,
    update_df_with_rare_codes
)

from utils import (
    ID_COLUMN,
    TEXT_COLUMN,
    TARGET_COLUMN,
    SUBJECT_ID_COLUMN,
    DOWNLOAD_DIRECTORY_MIMICIII,
    DOWNLOAD_DIRECTORY_MIMICIV,
    DOWNLOAD_DIRECTORY_MIMICIV_NOTE,
    DATA_DIRECTORY_PROCESSED
)



def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the codes dataframe"""
    df = df.rename(columns={"hadm_id": ID_COLUMN, "subject_id": SUBJECT_ID_COLUMN})
    df = df.dropna(subset=["icd_code"])
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, col="icd_code"))
        .reset_index()
    )
    return df

def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe"""
    df = df.rename(
        columns={
            "hadm_id": ID_COLUMN,
            "subject_id": SUBJECT_ID_COLUMN,
            "text": TEXT_COLUMN,
        }
    )
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT_COLUMN])
    return df


def get_d_dict(d, is_diag=True):
    dict_ = {}
    for i, row in d.iterrows():
        code = reformat_icd(row['icd_code'], version=row['icd_version'], is_diag=is_diag)
        dict_[code] = row['long_title']
    return dict_


random.seed(10)

logging.basicConfig(level=logging.INFO)
logging.info("Starting to process the MIMIC-IV data.")

# Load the data
download_dir_note = Path(DOWNLOAD_DIRECTORY_MIMICIV_NOTE)
download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIV)


mimic_proc = pd.read_csv(download_dir / "hosp/procedures_icd.csv.gz", dtype={"icd_code": str})
mimic_diag = pd.read_csv(download_dir / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})

mimic_d_proc = pd.read_csv(download_dir / "hosp/d_icd_procedures.csv.gz")
mimic_d_diag = pd.read_csv(download_dir / "hosp/d_icd_diagnoses.csv.gz")

mimic_notes = pd.read_csv(download_dir_note / "note/discharge.csv.gz")


# Format the codes by adding decimal points
mimic_proc["icd_code"] = mimic_proc.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=False
    ),
    axis=1,
)
mimic_diag["icd_code"] = mimic_diag.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=True
    ),
    axis=1,
)


# Process codes and notes
mimic_proc = parse_codes_dataframe(mimic_proc)
mimic_diag = parse_codes_dataframe(mimic_diag)
mimic_notes = parse_notes_dataframe(mimic_notes)

# Merge the codes and notes into a icd9 and icd10 dataframe
mimic_proc_9 = mimic_proc[mimic_proc["icd_version"] == 9]
mimic_proc_9 = mimic_proc_9.rename(columns={"icd_code": "icd9_proc"})
mimic_proc_10 = mimic_proc[mimic_proc["icd_version"] == 10]
mimic_proc_10 = mimic_proc_10.rename(columns={"icd_code": "icd10_proc"})

mimic_diag_9 = mimic_diag[mimic_diag["icd_version"] == 9]
mimic_diag_9 = mimic_diag_9.rename(columns={"icd_code": "icd9_diag"})
mimic_diag_10 = mimic_diag[mimic_diag["icd_version"] == 10]
mimic_diag_10 = mimic_diag_10.rename(columns={"icd_code": "icd10_diag"})

mimiciv_9 = mimic_notes.merge(
    mimic_proc_9[[ID_COLUMN, "icd9_proc"]], on=ID_COLUMN, how="left"
)
mimiciv_9 = mimiciv_9.merge(
    mimic_diag_9[[ID_COLUMN, "icd9_diag"]], on=ID_COLUMN, how="left"
)

mimiciv_10 = mimic_notes.merge(
    mimic_proc_10[[ID_COLUMN, "icd10_proc"]], on=ID_COLUMN, how="left"
)
mimiciv_10 = mimiciv_10.merge(
    mimic_diag_10[[ID_COLUMN, "icd10_diag"]], on=ID_COLUMN, how="left"
)

# remove notes with no codes
mimiciv_9 = mimiciv_9.dropna(subset=["icd9_proc", "icd9_diag"], how="all")
mimiciv_10 = mimiciv_10.dropna(subset=["icd10_proc", "icd10_diag"], how="all")

# convert NaNs to empty lists
mimiciv_9["icd9_proc"] = mimiciv_9["icd9_proc"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_9["icd9_diag"] = mimiciv_9["icd9_diag"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_10["icd10_proc"] = mimiciv_10["icd10_proc"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_10["icd10_diag"] = mimiciv_10["icd10_diag"].apply(
    lambda x: [] if x is np.nan else x
)



# get code descriptions
d_proc_9 = mimic_d_proc[mimic_d_proc.icd_version == 9]
d_proc_10 = mimic_d_proc[mimic_d_proc.icd_version == 10]

dict_proc_9 = get_d_dict(d_proc_9, is_diag=False)
dict_proc_10 = get_d_dict(d_proc_10, is_diag=False)

d_diag_9 = mimic_d_diag[mimic_d_diag.icd_version == 9]
d_diag_10 = mimic_d_diag[mimic_d_diag.icd_version == 10]

dict_diag_9 = get_d_dict(d_diag_9, is_diag=True)
dict_diag_10 = get_d_dict(d_diag_10, is_diag=True)

# merge dictionaries and sort by key
dict_9 = {**dict_proc_9, **dict_diag_9}
dict_10 = {**dict_proc_10, **dict_diag_10}

dict_9 = dict(sorted(dict_9.items()))
dict_10 = dict(sorted(dict_10.items()))

assert len(dict_9) == len(dict_proc_9) + len(dict_diag_9)
assert len(dict_10) == len(dict_proc_10) + len(dict_diag_10)


# process data
mimiciv_9 = mimiciv_9.copy()
mimiciv_10 = mimiciv_10.copy()

for col in ["icd9_proc", "icd9_diag"]:
    mimiciv_9[f'full_{col}'] = mimiciv_9[col]
for col in ["icd10_proc", "icd10_diag"]:
    mimiciv_10[f'full_{col}'] = mimiciv_10[col]

mimiciv_9 = filter_codes(mimiciv_9, ["icd9_proc", "icd9_diag"], min_count=10)
mimiciv_10 = filter_codes(mimiciv_10, ["icd10_proc", "icd10_diag"], min_count=10)

mimiciv_9[TARGET_COLUMN] = mimiciv_9["icd9_proc"] + mimiciv_9["icd9_diag"]
mimiciv_10[TARGET_COLUMN] = mimiciv_10["icd10_proc"] + mimiciv_10["icd10_diag"]

mimiciv_9[f'full_{TARGET_COLUMN}'] = mimiciv_9[f'full_icd9_proc'] + mimiciv_9[f'full_icd9_diag']
mimiciv_10[f'full_{TARGET_COLUMN}'] = mimiciv_10[f'full_icd10_proc'] + mimiciv_10[f'full_icd10_diag']

# remove empty target
mimiciv_9 = mimiciv_9[mimiciv_9[TARGET_COLUMN].apply(lambda x: len(x) > 0)]
mimiciv_10 = mimiciv_10[mimiciv_10[TARGET_COLUMN].apply(lambda x: len(x) > 0)]

# update the dataframe with rare codes
mimiciv_9 = update_df_with_rare_codes(mimiciv_9, target_column=TARGET_COLUMN)
mimiciv_10 = update_df_with_rare_codes(mimiciv_10, target_column=TARGET_COLUMN)

# reset index
mimiciv_9 = mimiciv_9.reset_index(drop=True)
mimiciv_10 = mimiciv_10.reset_index(drop=True)

# Text preprocess the notes
mimiciv_9 = preprocess_documents(df=mimiciv_9)
mimiciv_10 = preprocess_documents(df=mimiciv_10)

logging.info(f"""

Finished processing the MIMIC-IV data.
      
Number of all possible ICD-10 codes: {len(dict_10)}
Number of all possible ICD-9 codes: {len(dict_9)}

Number of full ICD-10 codes: {mimiciv_10[f'full_{TARGET_COLUMN}'].explode().nunique()}
Number of frequent ICD-10 codes: {mimiciv_10[TARGET_COLUMN].explode().nunique()}
Number of rare ICD-10 codes: {mimiciv_10['rare_' + TARGET_COLUMN].explode().nunique()}

Number of full ICD-9 codes: {mimiciv_9[f'full_{TARGET_COLUMN}'].explode().nunique()}
Number of frequent ICD-9 codes: {mimiciv_9[TARGET_COLUMN].explode().nunique()}
Number of rare ICD-9 codes: {mimiciv_9['rare_' + TARGET_COLUMN].explode().nunique()}

""")


# save files to disk
output_dir = Path(DATA_DIRECTORY_PROCESSED)
output_dir.mkdir(parents=True, exist_ok=True)

# reset index
mimiciv_9 = mimiciv_9.reset_index(drop=True)
mimiciv_10 = mimiciv_10.reset_index(drop=True)

mimiciv_9.to_feather(output_dir / "mimiciv_icd9.feather")
mimiciv_10.to_feather(output_dir / "mimiciv_icd10.feather")

dict_9 = {k: re.sub("[^A-Za-z0-9]+", " ", v) for k, v in dict_9.items()}
dict_10 = {k: re.sub("[^A-Za-z0-9]+", " ", v) for k, v in dict_10.items()}

# Check all full codes are in the description
assert all(code in dict_9 for code in mimiciv_9[f'full_{TARGET_COLUMN}'].explode().unique())
assert all(code in dict_10 for code in mimiciv_10[f'full_{TARGET_COLUMN}'].explode().unique())

description_all = {
    "icd9": dict_9,
    "icd10": dict_10,
}

with open(output_dir / "code_descriptions.pkl", "wb") as f:
    pickle.dump(description_all, f)




