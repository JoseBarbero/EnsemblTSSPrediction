import pandas as pd
import numpy as np
import pickle
import sys
import os

#DATAFRAME_CSV = '../rawdata/TSS/EveryEnsemblTranscript_pos_and_neg_TSS.csv'
DATAFRAME_CSV = sys.argv[1]
OUT_DIR = sys.argv[2]
ID = sys.argv[3]

TRAIN_CHROMOSOMES = {'3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '17', '18', '20', 'X', 'Y'}
VAL_CHROMOSOMES = {'16'}
TEST_CHROMOSOMES = {'1', '2', '19', '21'}

# X pos seqs
X_TRAIN_SEQS_POS_FILE = os.path.join(OUT_DIR, f"{ID}_X_train_TSSseqs_pos.txt")
X_VAL_SEQS_POS_FILE = os.path.join(OUT_DIR, f"{ID}_X_val_TSSseqs_pos.txt")
X_TEST_SEQS_POS_FILE = os.path.join(OUT_DIR, f"{ID}_X_test_TSSseqs_pos.txt")

# X neg seqs
X_TRAIN_SEQS_NEG_FILE = os.path.join(OUT_DIR, f"{ID}_X_train_TSSseqs_neg.txt")
X_VAL_SEQS_NEG_FILE = os.path.join(OUT_DIR, f"{ID}_X_val_TSSseqs_neg.txt")
X_TEST_SEQS_NEG_FILE = os.path.join(OUT_DIR, f"{ID}_X_test_TSSseqs_neg.txt")

# Processed X (as onehot) and y
X_TRAIN_FILE = os.path.join(OUT_DIR, f"{ID}_X_train_TSS.pkl")
X_VAL_FILE = os.path.join(OUT_DIR, f"{ID}_X_val_TSS.pkl")
X_TEST_FILE = os.path.join(OUT_DIR, f"{ID}_X_test_TSS.pkl")

Y_TRAIN_FILE = os.path.join(OUT_DIR, f"{ID}_y_train_TSS.pkl")
Y_VAL_FILE = os.path.join(OUT_DIR, f"{ID}_y_val_TSS.pkl")
Y_TEST_FILE = os.path.join(OUT_DIR, f"{ID}_y_test_TSS.pkl")


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
    pos_seqs = df['flankedTSS'].values.astype(str)
    pos_seqs = pos_seqs[[len(seq) == 1003 for seq in pos_seqs]]
    return pos_seqs

def get_neg_seqs(df, n_neg):

    neg_seqs_cols = [f'negative_instance_{n}' for n in range(n_neg)]

    neg_seqs = df[neg_seqs_cols].values.astype(str)
    neg_seqs = neg_seqs[neg_seqs != 'nan']

    return neg_seqs

def seq_to_onehot_array(seqs):
    X = []
    basetovalue = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    for seq in seqs:
        seq_values = []
        for base in seq:
            seq_values.append(basetovalue.get(base.upper(), [0, 0, 0, 0]))     # Aquí no estoy diferenciando entre caps y no, por lo tanto no diferencio entre secuencia 3' y 5'.
        X.append(seq_values)
    seq_len = len(X[0])
    X = np.array(X)
    print('Pre shape:', X.shape)
    X = np.reshape(X, (-1, seq_len, 4))
    print('Post shape:', X.shape)
    return X



df = pd.read_csv(DATAFRAME_CSV, dtype=str)

train_df, val_df, test_df = split_dataframe(df)


x_train_pos_seqs = get_pos_seqs(train_df)
x_val_pos_seqs = get_pos_seqs(val_df)
x_test_pos_seqs = get_pos_seqs(test_df)

x_train_neg_seqs = get_neg_seqs(train_df, 10)
x_val_neg_seqs = get_neg_seqs(val_df, 10)
x_test_neg_seqs = get_neg_seqs(test_df, 10)

y_train_pos = np.ones((x_train_pos_seqs.shape[0], 1))
y_val_pos = np.ones((x_val_pos_seqs.shape[0], 1))
y_test_pos = np.ones((x_test_pos_seqs.shape[0], 1))

y_train_neg = np.ones((x_train_neg_seqs.shape[0], 1))
y_val_neg = np.ones((x_val_neg_seqs.shape[0], 1))
y_test_neg = np.ones((x_test_neg_seqs.shape[0], 1))


# Save seqs to file

np.savetxt(X_TRAIN_SEQS_POS_FILE, x_train_pos_seqs, fmt="%s")
np.savetxt(X_VAL_SEQS_POS_FILE, x_val_pos_seqs, fmt="%s")
np.savetxt(X_TEST_SEQS_POS_FILE, x_test_pos_seqs, fmt="%s")

np.savetxt(X_TRAIN_SEQS_NEG_FILE, x_train_neg_seqs, fmt="%s")
np.savetxt(X_VAL_SEQS_NEG_FILE, x_val_neg_seqs, fmt="%s")
np.savetxt(X_TEST_SEQS_NEG_FILE, x_test_neg_seqs, fmt="%s")


# Save processed one-hot datasets with y included to pkl

X_train_pos = seq_to_onehot_array(x_train_pos_seqs)
X_val_pos = seq_to_onehot_array(x_val_pos_seqs)
X_test_pos = seq_to_onehot_array(x_test_pos_seqs)

X_train_neg = seq_to_onehot_array(x_train_neg_seqs)
X_val_neg = seq_to_onehot_array(x_val_neg_seqs)
X_test_neg = seq_to_onehot_array(x_test_neg_seqs)

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