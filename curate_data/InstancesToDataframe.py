from Bio import SeqIO
import pandas as pd

'''
This simple script gets transcripts from a fasta file and creates a structured dataframe with them.
'''


# This file has every transcript from hg38.p13 downloaded using Biomart from https://www.ensembl.org/
FASTA_FILE = '../rawdata/EveryEnsemblTranscript_withflanks.fasta'
DATAFRAME_OUT_FILE = '../rawdata/EveryEnsemblTranscript_withflanks_dataframe.csv'

# These fields come from the fasta file header *IN THIS SPECIFIC ORDER* (based on the fields selected from Biomart)
ensembl_df = pd.DataFrame(columns=['Gene stable ID version', 'Transcript stable ID version', 'Gene start (bp)', 'Chromosome_scaffold name', 
                                   'Gene name', 'Transcript start (bp)', 'Transcription start site (TSS)', 'CDS Length', 'CDS start', 'CDS end',
                                   'Transcript end (bp)', 'Start phase', 'End phase', 'cDNA coding start', 'cDNA coding end', 'Genomic coding start', 
                                   'Genomic coding end', 'Constitutive exon', 'Exon rank in transcript', 'Strand', 'Exon region start (bp)', 'Exon region end (bp)',
                                   'Exon_stable_ID', 'Sequence'])

ensembl_data = list(SeqIO.parse(FASTA_FILE, "fasta"))

i = 0
for record in ensembl_data:
    Gene_stable_ID_version, Transcript_stable_ID_version, Gene_start_bp, Chromosome_scaffold_name, Gene_name, Transcript_start_bp, \
    Transcription_start_site_TSS, CDS_Length, CDS_start, CDS_end, Transcript_end_bp, Start_phase, End_phase, \
    cDNA_coding_start, cDNA_coding_end, Genomic_coding_start, Genomic_coding_end, Constitutive_exon, Exon_rank_in_transcript, Strand, \
    Exon_region_start_bp, Exon_region_end_bp, Exon_stable_ID = record.id.split('|')
    
    ensembl_df.loc[i] = [Gene_stable_ID_version, Transcript_stable_ID_version, Gene_start_bp, Chromosome_scaffold_name, Gene_name, Transcript_start_bp, \
                        Transcription_start_site_TSS, CDS_Length, CDS_start, CDS_end, Transcript_end_bp, Start_phase, End_phase, \
                        cDNA_coding_start, cDNA_coding_end, Genomic_coding_start, Genomic_coding_end, Constitutive_exon, Exon_rank_in_transcript, Strand, \
                        Exon_region_start_bp, Exon_region_end_bp, Exon_stable_ID, str(record.seq)]
    i += 1
    print(f'{i}/{len(ensembl_data)}', end='\r', flush=True)


ensembl_df.to_csv(DATAFRAME_OUT_FILE)