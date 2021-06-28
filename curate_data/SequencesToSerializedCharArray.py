import sys
sys.path.append("../utils")
from ReadData import seqfile_to_instances
import numpy as np
import pickle

def sequences_to_char_array(seqs_array):
    # Pasa de un array con cadenas a un array de arrays de caracteres
    return np.array([np.array(list(seq)) for seq in seqs_array])


X_train_seqs_pos = seqfile_to_instances('../data/TSS/seqs/X_train_TSSseqs_pos.txt')
X_train_seqs_neg = seqfile_to_instances('../data/TSS/seqs/X_train_TSSseqs_neg.txt')
X_val_seqs_pos = seqfile_to_instances('../data/TSS/seqs/X_val_TSSseqs_pos.txt')
X_val_seqs_neg = seqfile_to_instances('../data/TSS/seqs/X_val_TSSseqs_neg.txt')
X_test_seqs_pos = seqfile_to_instances('../data/TSS/seqs/X_test_TSSseqs_pos.txt')
X_test_seqs_neg = seqfile_to_instances('../data/TSS/seqs/X_test_TSSseqs_neg.txt')

X_train_seqs_pos = sequences_to_char_array(X_train_seqs_pos)
X_train_seqs_neg = sequences_to_char_array(X_train_seqs_neg)
X_val_seqs_pos = sequences_to_char_array(X_val_seqs_pos)
X_val_seqs_neg = sequences_to_char_array(X_val_seqs_neg)
X_test_seqs_pos = sequences_to_char_array(X_test_seqs_pos)
X_test_seqs_neg = sequences_to_char_array(X_test_seqs_neg)

# X pos seqs
X_TRAIN_SEQS_POS_FILE = '../data/TSS/seqs/X_train_TSSseqs_pos_chararray.txt'
X_VAL_SEQS_POS_FILE = '../data/TSS/seqs/X_val_TSSseqs_pos_chararray.txt'
X_TEST_SEQS_POS_FILE = '../data/TSS/seqs/X_test_TSSseqs_pos_chararray.txt'
# X neg seqs
X_TRAIN_SEQS_NEG_FILE = '../data/TSS/seqs/X_train_TSSseqs_neg_chararray.txt'
X_VAL_SEQS_NEG_FILE = '../data/TSS/seqs/X_val_TSSseqs_neg_chararray.txt'
X_TEST_SEQS_NEG_FILE = '../data/TSS/seqs/X_test_TSSseqs_neg_chararray.txt'

X_train_pos_file = open(X_TRAIN_SEQS_POS_FILE, 'wb')
pickle.dump(X_train_seqs_pos, X_train_pos_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_pos_file.close()

X_val_pos_file = open(X_VAL_SEQS_POS_FILE, 'wb')
pickle.dump(X_val_seqs_pos, X_val_pos_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_val_pos_file.close()

X_test_pos_file = open(X_TEST_SEQS_POS_FILE, 'wb')
pickle.dump(X_test_seqs_pos, X_test_pos_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_test_pos_file.close()

X_train_neg_file = open(X_TRAIN_SEQS_NEG_FILE, 'wb')
pickle.dump(X_train_seqs_neg, X_train_neg_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_neg_file.close()

X_val_neg_file = open(X_VAL_SEQS_NEG_FILE, 'wb')
pickle.dump(X_val_seqs_neg, X_val_neg_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_val_neg_file.close()

X_test_neg_file = open(X_TEST_SEQS_NEG_FILE, 'wb')
pickle.dump(X_test_seqs_neg, X_test_neg_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_test_neg_file.close()