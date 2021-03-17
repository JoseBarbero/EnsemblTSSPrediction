import pandas as pd
import numpy as np
import re
import random

'''
This script gets a dataframe (created by GetTISFlanks.py) and generates negative ones.
To simulate negative instances we will select POS_TO_NEG_RATIO random positions in a transcript where the real TIS codon is found.

Usually we find 10 negative instances to every positive one, so this a good value to start with.
We keep those 10 negative instances in a new column of the dataframe as a list. This is the easiest way to allow filtering these instances by chromosome, gene, etc.
'''


POS_INSTANCES_DF = '../EveryEnsemblTranscript_withTIS_dataframe.csv'
#POS_INSTANCES_DF = '../Test_withTIS_dataframe.csv'
POS_TO_NEG_RATIO = 10   # We generate 10 negative instances for every positive one
DATAFRAME_OUT_FILE = '../EveryEnsemblTranscript_pos_and_neg.csv'


ensembl_df = pd.read_csv(POS_INSTANCES_DF)

for i in range(POS_TO_NEG_RATIO):
    ensembl_df[f'negative_instance_{i}'] = ''


i=0

for idx, row in ensembl_df.iterrows():
    tis_flank_length = row['TISFlankLength']
    tis_codon = row['TIScodon']

    transcript_flank_length = row['transcriptFlankLength']
    transcript = row['Sequence']
    clean_transcript = transcript[transcript_flank_length:-transcript_flank_length]

    # Get every occurence of tis_codon in that transcript
    occurrences_idxs = [i.start() for i in re.finditer('(?='+tis_codon+')', clean_transcript)]

    # Get a random selection of POS_TO_NEG_RATIO occurrences
    # Get 1 less every time there are no enough occurrences    
    local_pos_to_neg = POS_TO_NEG_RATIO
    while True:
        try:
            neg_idxs = random.sample(occurrences_idxs, local_pos_to_neg)
        except ValueError:
            local_pos_to_neg -= 1
        else:
            break


    j = 0
    for neg_idx in neg_idxs:
        idx_with_margin = neg_idx + row['transcriptFlankLength']    # To get the actual coordinate in the flanked transcript
        ensembl_df.at[idx, f'negative_instance_{i}'] = transcript[idx_with_margin-100:idx_with_margin+103]
        i+=j
    
    i +=1
    print(f'{i}/{len(ensembl_df)}', end='\r', flush=True)


ensembl_df.to_csv(DATAFRAME_OUT_FILE)