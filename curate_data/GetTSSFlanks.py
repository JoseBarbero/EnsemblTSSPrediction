from Bio import SeqIO
import pandas as pd
import numpy as np


'''
This script gets a dataframe (created by InstancesToDataframe.py) and adds 4 new fields:
    - flankedTSS: TSS codon flanked by TSS_FLANK_LENGTH bases
    - TSScodon: the 3 bases that start the translation
    - transcriptFlankLength
    - TSSFlankLength
'''


#DATAFRAME_CSV = 'EveryEnsemblTranscript_dataframe.csv'
#FLANK_LENGTH = 0
DATAFRAME_CSV = '../rawdata/EveryEnsemblTranscript_withflanks_dataframe.csv'
DATAFRAME_OUT_FILE = '../rawdata/TSS/EveryEnsemblTranscript_withTSS_dataframe.csv'

# The sequence from Ensemble comes with 1000 bases before and after the transcript
UPSTREAM_FLANK_LENGTH = 1000
DOWNSTREAM_FLANK_LENGTH = 1000
UPSTREAM_TSS_FLANK_LENGTH = 700
DOWNSTREAM_TSS_FLANK_LENGTH = 300


ensembl_df = pd.read_csv(DATAFRAME_CSV)
ensembl_df['transcriptFlankLength_Upstream'] = UPSTREAM_FLANK_LENGTH
ensembl_df['transcriptFlankLength_Downstream'] = DOWNSTREAM_FLANK_LENGTH
ensembl_df['flankedTSS'] = ''
ensembl_df['TSScodon'] = ''
ensembl_df['TSScodonIndexUnflankedTranscript'] = None
ensembl_df['TSSFlankLengthUpstream'] = UPSTREAM_TSS_FLANK_LENGTH
ensembl_df['TSSFlankLengthDownstream'] = DOWNSTREAM_TSS_FLANK_LENGTH


print('RAW len', len(ensembl_df))
ensembl_df = ensembl_df[ensembl_df['Genomic coding start'].notna()] # Aquí se pierden como la mitad de las instancias ¿Por qué faltan tantas?
print('CLEAN len', len(ensembl_df))

i = 0
for idx, transcript in ensembl_df.iterrows():

    seq = transcript['Sequence']
    if transcript['Strand'] == 1:

        tss_idx = 0 + UPSTREAM_FLANK_LENGTH
        flankedTSS = seq[tss_idx-UPSTREAM_TSS_FLANK_LENGTH:tss_idx+DOWNSTREAM_TSS_FLANK_LENGTH+3]
        ensembl_df.at[idx, 'flankedTSS'] = flankedTSS
        ensembl_df.at[idx, 'TSScodon'] = seq[tss_idx:tss_idx+3]

    elif transcript['Strand'] == 1:

        tss_idx = 0 + DOWNSTREAM_FLANK_LENGTH
        flankedTSS = seq[tss_idx-DOWNSTREAM_TSS_FLANK_LENGTH:tss_idx+UPSTREAM_TSS_FLANK_LENGTH+3]
        ensembl_df.at[idx, 'flankedTSS'] = flankedTSS
        ensembl_df.at[idx, 'TSScodon'] = seq[tss_idx:tss_idx+3]

    i += 1
    print(f'{i}/{len(ensembl_df)}', end='\r', flush=True)

ensembl_df.to_csv(DATAFRAME_OUT_FILE)