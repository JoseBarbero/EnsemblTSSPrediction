import pandas as pd
import numpy as np
import re
import random

'''
This script gets a dataframe (created by GetTSSFlanks.py) and generates negative ones.
To simulate negative instances we will select POS_TO_NEG_RATIO random positions in a transcript where the real TSS codon is found.

Usually we find 10 negative instances to every positive one, so this a good value to start with.
We keep those 10 negative instances in a new column of the dataframe as a list. This is the easiest way to allow filtering these instances by chromosome, gene, etc.
'''


POS_INSTANCES_DF = '../rawdata/TSS/EveryEnsemblTranscript_withTSS_dataframe.csv'
POS_TO_NEG_RATIO = 1   # We generate 1 negative instance for every positive one
DATAFRAME_OUT_FILE = '../rawdata/TSS/EveryEnsemblTranscript_pos_and_neg_TSS_1to1.csv'


ensembl_df = pd.read_csv(POS_INSTANCES_DF)

for i in range(POS_TO_NEG_RATIO):
    ensembl_df[f'negative_instance_{i}'] = ''


i=0

for idx, row in ensembl_df.iterrows():
    tss_flank_length_upstream = row['TSSFlankLengthUpstream']
    tss_flank_length_downstream = row['TSSFlankLengthDownstream']

    tss_codon = row['TSScodon']

    transcript_flank_length_upstream = row['transcriptFlankLengthUpstream']
    transcript_flank_length_downstream = row['transcriptFlankLengthDownstream']

    transcript = row['Sequence']

    if row['Strand'] == 1:
        clean_transcript = transcript[transcript_flank_length_upstream:-transcript_flank_length_downstream]
    elif row['Strand'] == -1:
        clean_transcript = transcript[transcript_flank_length_downstream:-transcript_flank_length_upstream]

    # Get every occurence of tss_codon in that transcript
    occurrences_idxs = [ocurr.start() for ocurr in re.finditer('(?='+tss_codon+')', clean_transcript)]

    # Remove the actual TSS to avoid getting it as a negative instance
    tss_idx = 0
    if tss_idx in occurrences_idxs:
        occurrences_idxs.remove(tss_idx) 

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

        if row['Strand'] == 1:
            idx_with_margin = neg_idx + tss_flank_length_upstream    # To get the actual coordinate in the flanked transcript
            flanked_fake_tss = transcript[idx_with_margin-tss_flank_length_upstream:idx_with_margin+tss_flank_length_downstream+3]
        elif row['Strand'] == -1:
            idx_with_margin = neg_idx + tss_flank_length_downstream    # To get the actual coordinate in the flanked transcript
            flanked_fake_tss = transcript[idx_with_margin-tss_flank_length_downstream:idx_with_margin+tss_flank_length_upstream+3]
        
        # Maybe that random codon is a real TSS in another row
        # I tried this and never happened in 1000000 cases
        # It makes the processing much slower so I'll skip it
        # if flanked_fake_tss in ensembl_df['flankedTSS'].values:
        #     
        #     print('\n\n\nFALSE NEGATIVE!!\n\n\n')
        #     print(flanked_fake_tss)
        #     print(row['flankedTSS'])
        #     input()
        
        ensembl_df.at[idx, f'negative_instance_{j}'] = flanked_fake_tss
        
        j+=1
    
    i +=1
    print(f'{i}/{len(ensembl_df)}', end='\r', flush=True)


ensembl_df.to_csv(DATAFRAME_OUT_FILE)