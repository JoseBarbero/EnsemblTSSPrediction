from Bio import SeqIO
import pandas as pd
import numpy as np


'''
This script gets a dataframe (created by InstancesToDataframe.py) and adds 4 new fields:
    - flankedTIS: TIS codon flanked by TIS_FLANK_LENGTH bases
    - TIScodon: the 3 bases that start the translation
    - transcriptFlankLength
    - TISFlankLength
'''


#DATAFRAME_CSV = 'EveryEnsemblTranscript_dataframe.csv'
#FLANK_LENGTH = 0
DATAFRAME_CSV = '../rawdata/EveryEnsemblTranscript_withflanks_dataframe.csv'
DATAFRAME_OUT_FILE = '../rawdata/TIS/EveryEnsemblTranscript_withTIS_dataframe.csv'

# The sequence from Ensemble comes with 1000 bases before and after the transcript
UPSTREAM_FLANK_LENGTH = 1000
DOWNSTREAM_FLANK_LENGTH = 1000
UPSTREAM_TIS_FLANK_LENGTH = 700
DOWNSTREAM_TIS_FLANK_LENGTH = 300


ensembl_df = pd.read_csv(DATAFRAME_CSV)
ensembl_df['transcriptFlankLengthUpstream'] = UPSTREAM_FLANK_LENGTH
ensembl_df['transcriptFlankLengthDownstream'] = DOWNSTREAM_FLANK_LENGTH
ensembl_df['flankedTIS'] = ''
ensembl_df['TIScodon'] = ''
ensembl_df['TIScodonIndexUnflankedTranscript'] = None
ensembl_df['TISFlankLengthUpstream'] = UPSTREAM_TIS_FLANK_LENGTH
ensembl_df['TISFlankLengthDownstream'] = DOWNSTREAM_TIS_FLANK_LENGTH


print('RAW len', len(ensembl_df))
ensembl_df = ensembl_df[ensembl_df['Genomic coding start'].notna()] # Aquí se pierden como la mitad de las instancias ¿Por qué faltan tantas?
print('CLEAN len', len(ensembl_df))

i = 0
for idx, transcript in ensembl_df.iterrows():

    seq = transcript['Sequence']
    #print('\n\n'+transcript['Transcript stable ID version'], 'Strand', transcript['Strand'])
    
    cdna_starts = [int(a) for a in transcript['Genomic coding start'].split(';')]
    cdna_stops = [int(a) for a in transcript['Genomic coding end'].split(';')]

    cdna_relative_starts = sorted([int(a) - int(transcript['Transcription start site (TSS)']) for a in cdna_starts])
    cdna_relative_stops = sorted([int(a) - int(transcript['Transcription start site (TSS)']) for a in cdna_stops])

    if transcript['Strand'] == 1:
        cds_start = int(cdna_relative_starts[0]) + UPSTREAM_FLANK_LENGTH
        #print(str(cds_start) + ' ' + str(transcript['Strand']), seq[cds_start:cds_start+3] == 'ATG', seq[cds_start:cds_start+3])
        flankedTIS = seq[cds_start-UPSTREAM_TIS_FLANK_LENGTH:cds_start+DOWNSTREAM_TIS_FLANK_LENGTH+3]
        ensembl_df.at[idx, 'flankedTIS'] = flankedTIS
        ensembl_df.at[idx, 'TIScodon'] = seq[cds_start:cds_start+3]
        ensembl_df.at[idx, 'TIScodonIndexUnflankedTranscript'] = cds_start - UPSTREAM_FLANK_LENGTH

    elif transcript['Strand'] == -1:
        cds_stop = abs(int(cdna_relative_stops[-1])) + DOWNSTREAM_FLANK_LENGTH
        #print(str(cds_stop) + ' ' + str(transcript['Strand']), seq[cds_stop:cds_stop+3] == 'ATG', seq[cds_stop:cds_stop+3])
        flankedTIS = seq[cds_stop-DOWNSTREAM_TIS_FLANK_LENGTH:cds_stop+UPSTREAM_TIS_FLANK_LENGTH+3]
        ensembl_df.at[idx, 'flankedTIS'] = flankedTIS
        ensembl_df.at[idx, 'TIScodon'] = seq[cds_stop:cds_stop+3]
        ensembl_df.at[idx, 'TIScodonIndexUnflankedTranscript'] = cds_stop - DOWNSTREAM_FLANK_LENGTH
    
    i += 1
    print(f'{i}/{len(ensembl_df)}', end='\r', flush=True)

ensembl_df.to_csv(DATAFRAME_OUT_FILE)