import pandas as pd
import numpy as np
import pickle

DATAFRAME_CSV = '../rawdata/TIS/EveryEnsemblTranscript_pos_and_neg_TIS.csv'

TRAIN_CHROMOSOMES = {'3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '17', '18', '20', 'X', 'Y'}
VAL_CHROMOSOMES = {'16'}
TEST_CHROMOSOMES = {'1', '2', '19', '21'}

# X pos seqs
X_TRAIN_SEQS_POS_FILE = '../data/TIS/seqs/X_train_TISseqs_pos.txt'
X_VAL_SEQS_POS_FILE = '../data/TIS/seqs/X_val_TISseqs_pos.txt'
X_TEST_SEQS_POS_FILE = '../data/TIS/seqs/X_test_TISseqs_pos.txt'
# X neg seqs
X_TRAIN_SEQS_NEG_FILE = '../data/TIS/seqs/X_train_TISseqs_neg.txt'
X_VAL_SEQS_NEG_FILE = '../data/TIS/seqs/X_val_TISseqs_neg.txt'
X_TEST_SEQS_NEG_FILE = '../data/TIS/seqs/X_test_TISseqs_neg.txt'

# Processed X (as onehot) and y
X_TRAIN_FILE = '../data/TIS/onehot_serialized/X_train_TIS.pkl'
X_VAL_FILE = '../data/TIS/onehot_serialized/X_val_TIS.pkl'
X_TEST_FILE = '../data/TIS/onehot_serialized/X_test_TIS.pkl'

Y_TRAIN_FILE = '../data/TIS/onehot_serialized/y_train_TIS.pkl'
Y_VAL_FILE = '../data/TIS/onehot_serialized/y_val_TIS.pkl'
Y_TEST_FILE = '../data/TIS/onehot_serialized/y_test_TIS.pkl'




# Qué pasa con los cromosomas raros tipo: CHR_HSCHR17_1_CTG1 ??
# Para verlos: print(df['Chromosome_scaffold name'].unique())
# Sin ellos me quedo con 102488 instancias de 115262

# Partir train/val/test
def split_dataframe(df):
    total = len(df)
    
    train = df[df['Chromosome_scaffold name'].isin(TRAIN_CHROMOSOMES)]
    val = df[df['Chromosome_scaffold name'].isin(VAL_CHROMOSOMES)]
    test = df[df['Chromosome_scaffold name'].isin(TEST_CHROMOSOMES)]
    
    return train, val, test


# Sacar las secuencias de cada línea del DF
def get_pos_seqs(df):
    pos_seqs = df['flankedTIS'].to_numpy(dtype=str)
    return pos_seqs

def get_neg_seqs(df, n_neg):

    neg_seqs_cols = [f'negative_instance_{n}' for n in range(n_neg)]

    neg_seqs = df[neg_seqs_cols].to_numpy(dtype=str)
    neg_seqs = np.reshape(neg_seqs, (-1, 1))
    neg_seqs = neg_seqs[neg_seqs != 'nan']
    neg_seqs = np.reshape(neg_seqs, (-1, 1))

    return neg_seqs

def seq_to_onehot_array(seqfile):
    X_fw = []
    X_rv = []
    basetovalue_fw = {'A': np.array([1, 0, 0, 0]), 'T': np.array([0, 1, 0, 0]), 'G': np.array([0, 0, 1, 0]), 'C': np.array([0, 0, 0, 1])}
    basetovalue_rv = {'A': basetovalue_fw['T'], 'T': basetovalue_fw['A'], 'G': basetovalue_fw['C'], 'C': basetovalue_fw['G']}
    with open(seqfile, "r") as _file:
        for line in _file:
            line_values_fw = []
            line_values_rv = []
            for base in line[:-1]:
                line_values_fw.append(basetovalue_fw.get(base.upper(), [0, 0, 0, 0]))     # Aquí no estoy diferenciando entre caps y no, por lo tanto no diferencio entre secuencia 3' y 5'.
                line_values_rv.append(basetovalue_rv.get(base.upper(), [0, 0, 0, 0]))
            X_fw.append(line_values_fw)
            X_rv.append(line_values_rv)
    return np.asarray(X_fw), np.asarray(X_rv)



df = pd.read_csv(DATAFRAME_CSV, dtype=str)

train_df, val_df, test_df = split_dataframe(df)


x_train_pos = get_pos_seqs(train_df)
x_val_pos = get_pos_seqs(val_df)
x_test_pos = get_pos_seqs(test_df)

x_train_neg = get_neg_seqs(train_df, 10)
x_val_neg = get_neg_seqs(val_df, 10)
x_test_neg = get_neg_seqs(test_df, 10)


y_train_pos = np.ones((x_train_pos.shape[0], 1))
y_val_pos = np.ones((x_val_pos.shape[0], 1))
y_test_pos = np.ones((x_test_pos.shape[0], 1))

y_train_neg = np.ones((x_train_neg.shape[0], 1))
y_val_neg = np.ones((x_val_neg.shape[0], 1))
y_test_neg = np.ones((x_test_neg.shape[0], 1))


# Save seqs to file

np.savetxt(X_TRAIN_SEQS_POS_FILE, x_train_pos, fmt="%s")
np.savetxt(X_VAL_SEQS_POS_FILE, x_val_pos, fmt="%s")
np.savetxt(X_TEST_SEQS_POS_FILE, x_test_pos, fmt="%s")

np.savetxt(X_TRAIN_SEQS_NEG_FILE, x_train_neg, fmt="%s")
np.savetxt(X_VAL_SEQS_NEG_FILE, x_val_neg, fmt="%s")
np.savetxt(X_TEST_SEQS_NEG_FILE, x_test_neg, fmt="%s")


# Save processed one-hot datasets with y included to pkl

X_train_pos = seq_to_onehot_array(X_TRAIN_SEQS_POS_FILE)[0]
X_val_pos = seq_to_onehot_array(X_VAL_SEQS_POS_FILE)[0]
X_test_pos = seq_to_onehot_array(X_TEST_SEQS_POS_FILE)[0]

X_train_neg = seq_to_onehot_array(X_TRAIN_SEQS_NEG_FILE)[0]
X_val_neg = seq_to_onehot_array(X_VAL_SEQS_NEG_FILE)[0]
X_test_neg = seq_to_onehot_array(X_TEST_SEQS_NEG_FILE)[0]


y_train_pos = np.ones((X_train_pos.shape[0]))
y_val_pos = np.ones((X_val_pos.shape[0]))
y_test_pos = np.ones((X_test_pos.shape[0]))

y_train_neg = np.zeros((X_train_neg.shape[0]))
y_val_neg = np.zeros((X_val_neg.shape[0]))
y_test_neg = np.zeros((X_test_neg.shape[0]))


X_train = np.concatenate((X_train_pos, X_train_neg))
X_val = np.concatenate((X_val_pos, X_val_neg))
X_test = np.concatenate((X_test_pos, X_test_neg))

y_train = np.concatenate((y_train_pos, y_train_neg))
y_val = np.concatenate((y_val_pos, y_val_neg))
y_test = np.concatenate((y_test_pos, y_test_neg))

print('X_train', X_train.shape)
print('X_val', X_val.shape)
print('X_test', X_test.shape)

print('y_train', y_train.shape)
print('y_val', y_val.shape)
print('y_test', y_test.shape)


X_train_file = open(X_TRAIN_FILE, 'wb')
pickle.dump(X_train, X_train_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_file.close()

X_val_file = open(X_VAL_FILE, 'wb')
pickle.dump(X_val, X_val_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_val_file.close()

X_test_file = open(X_TEST_FILE, 'wb')
pickle.dump(X_test, X_test_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_test_file.close()


y_train_file = open(Y_TRAIN_FILE, 'wb')
pickle.dump(y_train, y_train_file, protocol=4)
y_train_file.close()

y_val_file = open(Y_VAL_FILE, 'wb')
pickle.dump(y_val, y_val_file, protocol=4)
y_val_file.close()

y_test_file = open(Y_TEST_FILE, 'wb')
pickle.dump(y_test, y_test_file, protocol=4)
y_test_file.close()