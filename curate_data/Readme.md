1. Download the data from Ensembl Biomart.
2. InstancesToDataframe.py
3. GetTSSFlanks.py
4. GetTSSNegativeInstances.py
5. CreateTISPartitions.py


* Fields in Biomart have to be selected in this specific order: 'Gene stable ID version', 'Transcript stable ID version', 'Gene start (bp)', 'Chromosome_scaffold name', 'Gene name', 'Transcript start (bp)', 'Transcription start site (TSS)', 'CDS Length', 'CDS start', 'CDS end', 'Transcript end (bp)', 'Start phase', 'End phase', 'cDNA coding start', 'cDNA coding end', 'Genomic coding start', 'Genomic coding end', 'Constitutive exon', 'Exon rank in transcript', 'Strand', 'Exon region start (bp)', 'Exon region end (bp)', 'Exon_stable_ID'
