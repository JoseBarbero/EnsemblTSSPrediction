import pickle 
import numpy as np
from numpy import random
from fastsk_utils import FastaUtility

# Read data
X_train_seqs_pos_file = open('../data/TSS/seqs/X_train_TSSseqs_pos_chararray.txt', 'rb')
X_train_seqs_neg_file = open('../data/TSS/seqs/X_train_TSSseqs_neg_chararray.txt', 'rb')
X_test_seqs_pos_file = open('../data/TSS/seqs/X_test_TSSseqs_pos_chararray.txt', 'rb')
X_test_seqs_neg_file = open('../data/TSS/seqs/X_test_TSSseqs_neg_chararray.txt', 'rb')

X_train_seqs_pos = pickle.load(X_train_seqs_pos_file)
X_train_seqs_neg = pickle.load(X_train_seqs_neg_file)
X_test_seqs_pos = pickle.load(X_test_seqs_pos_file)
X_test_seqs_neg = pickle.load(X_test_seqs_neg_file)

X_train_seqs_pos_file.close()
X_train_seqs_neg_file.close()
X_test_seqs_pos_file.close()
X_test_seqs_neg_file.close()

print(X_train_seqs_pos.shape)
print(X_train_seqs_neg.shape)

X_train = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])
X_test = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])


# Shuffle X_train and y_train keeping the same order
random.seed(42)
idx = random.choice(X_train.shape[0], X_train.shape[0], replace=False)
X_train = X_train[idx]
y_train = y_train[idx]

# Shuffle X_test and y_test keeping the same order
idx = random.choice(X_test.shape[0], X_test.shape[0], replace=False)
X_test = X_test[idx]
y_test = y_test[idx]

fasta_train_file = '../data/TSS/fastas/X_train_TSS.fasta'
fasta_test_file = '../data/TSS/fastas/X_test_TSS.fasta'
fasta_train_file_p = open(fasta_train_file, 'w')
fasta_test_file_p = open(fasta_test_file, 'w')

for i in range(X_train.shape[0]):
    seq = X_train[i].tostring().decode('utf-8')
    seq_label = y_train[i]
    fasta_train_file_p.write('>' + str(seq_label) + '\n')
    fasta_train_file_p.write(str(seq) + '\n')

for i in range(X_test.shape[0]):
    seq = X_test[i].tostring().decode('utf-8')
    seq_label = y_test[i]
    fasta_test_file_p.write('>' + str(seq_label) + '\n')
    fasta_test_file_p.write(str(seq) + '\n')

fasta_train_file_p.close()
fasta_test_file_p.close()

fasta_train_file = '../data/TSS/fastas/X_train_TSS.fasta'
fasta_test_file = '../data/TSS/fastas/X_test_TSS.fasta'

fastk_train_output_file_X = '../data/TSS/fastsk_format/X_train_TSS.pkl'
fastk_train_output_file_y = '../data/TSS/fastsk_format/y_train_TSS.pkl'
fastk_test_output_file_X = '../data/TSS/fastsk_format/X_test_TSS.pkl'
fastk_test_output_file_y = '../data/TSS/fastsk_format/y_test_TSS.pkl'

### Read the data
reader = FastaUtility()
X_train, y_train = reader.read_data(fasta_train_file)
X_test, y_test = reader.read_data(fasta_test_file)
#y_test = np.array(y_test).reshape(-1, 1)

# Save the data
with open(fastk_train_output_file_X, 'wb') as f:
    pickle.dump(X_train, f)

with open(fastk_train_output_file_y, 'wb') as f:
    pickle.dump(y_train, f)

with open(fastk_test_output_file_X, 'wb') as f:
    pickle.dump(X_test, f)

with open(fastk_test_output_file_y, 'wb') as f:
    pickle.dump(y_test, f)


