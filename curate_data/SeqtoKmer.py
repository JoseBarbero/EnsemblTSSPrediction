import sys

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def seqfile_to_kmerfile(seqfile, kmerfile, k, label):
    with open(seqfile, 'r') as sf:
        with open(kmerfile, 'w') as kf:
            kf.write(f"sequence\tlabel\n")
            for seq in sf.readlines():
                kmers = seq2kmer(seq, k)
                kf.write(f"{kmers[:-1]}\t{label}\n")
                

if __name__ == "__main__":
    
    seqfile = sys.argv[1]
    k = int(sys.argv[2])
    label = int(sys.argv[3])
    outfile = sys.argv[4]

    seqfile_to_kmerfile(seqfile, outfile, k, label)


    