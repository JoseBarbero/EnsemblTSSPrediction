import sys
sys.path.append("../utils")
from ReadData import seqfile_to_instances
import numpy as np
import pickle as pkl


def seqs_to_onehot_array(seqs):
    X = []
    basetovalue = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    for seq in seqs:
        seq_values = []
        for base in seq:
            seq_values.append(basetovalue.get(base.upper(), [0, 0, 0, 0]))     # Aquí no estoy diferenciando entre caps y no, por lo tanto no diferencio entre secuencia 3' y 5'.
        X.append(seq_values)
    seq_len = len(X[0])
    X = np.array(X)
    X = np.reshape(X, (-1, seq_len, 4))
    return X


X_train_seqs = pkl.load(open('../data/TSS/seqs/mouse_X_train_TSS.pkl', 'rb'))
X_val_seqs = pkl.load(open('../data/TSS/seqs/mouse_X_val_TSS.pkl', 'rb'))
X_test_seqs = pkl.load(open('../data/TSS/seqs/mouse_X_test_TSS.pkl', 'rb'))

# X pos seqs
X_TRAIN_ONEHOT_FILE = '../data/TSS/onehot_serialized/mouse_X_train_TSS.pkl'
X_VAL_ONEHOT_FILE = '../data/TSS/onehot_serialized/mouse_X_val_TSS.pkl'
X_TEST_ONEHOT_FILE = '../data/TSS/onehot_serialized/mouse_X_test_TSS.pkl'

X_train_file = open(X_TRAIN_ONEHOT_FILE, 'wb')
pkl.dump(X_train_seqs, X_train_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_file.close()

X_val_file = open(X_VAL_ONEHOT_FILE, 'wb')
pkl.dump(X_val_seqs, X_val_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_val_file.close()

X_test_file = open(X_TEST_ONEHOT_FILE, 'wb')
pkl.dump(X_test_seqs, X_test_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_test_file.close()