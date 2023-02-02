import sys
import os
sys.path.append("../utils")
import time
from datetime import timedelta
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import classification_report  # classfication summary
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pickle 


def distribution_preserving_partitioning(D, N, C, P, K):
    """
    Divides the dataset into pieces that maintain the
    same statistical properties.

    Args:
    D: dataset
    N: number of instance (vectors) in D
    (d: dimensions) -> deleted, not used
    C: number of classes
    P: number of partitions
    K: number of clusters
    (U: random indexes) -> deleted, used inside the function

    Returns:
    Dp: partitioned dataset
    """
    Dc = group_classwise(D, C)
    Dp = [[] for p in range(P)]
    
    for c in range(C):
        # DcK = dataset of class c clustered into K clusters
        DcK = MiniBatchKMeans(n_clusters=K, random_state=0).fit(Dc[c][:, :-1]) # KMeans causes "core dumped")
        for k in range(K):
            # Dck = dataset of class c and cluster k
            Dck = Dc[c][DcK.labels_ == k]
            # Nck = number of instances in class c and cluster k
            Nck = Dck.shape[0]
            for p in range(P):
                # Nckp = number of instances in class c, cluster k and partition p rounded up
                Nckp = int(np.ceil(Nck/P))
                U_idxs = random.choice(Nck, Nckp, replace=False)
                # Dckp = dataset of class c, cluster k and partition p
                Dckp = Dck[U_idxs]
                Dp[p].extend(Dckp)
                Dck = np.delete(Dck, U_idxs, axis=0)
                Nck -= Nckp
    return [np.array(Dp[p]) for p in range(P)]


def dip_svm_learning(D_onehot, N, P, K, Tgamma, params):
    """
    Performs the DIP-SVM learning algorithm.

    Args:
    D_onehot: dataset in one-hot encoding (kmeans can't handle strings that is why we need one-hot encoding to get the indexes)
    D_string: dataset in string encoding
    N: number of instance (vectors) in D
    P: number of partitions
    K: number of clusters
    Tgamma: threshold gamma
    params: parameters of the SVM
        kernel_function: kernel function
        kernel_parameters: kernel parameters

    Returns:
    S: support vectors
    alpha: lagrangian multipliers
    """
    
    Dp = distribution_preserving_partitioning(D_onehot, N, C, P, K)
    
    l = 0 # Level
    alpha = 0
    S = []
    Sl = []
    
    next_level_D = [[] for p in range(P)]
    next_level_alpha = []
    while True:  
        
        Slp = []
        for p in range(P):  # TODO Parallelize
            # Train the SVM
            print("Partition: ", p)
            print("\tDp len", len(Dp))
            print("\tDp[p]: ", Dp[p].shape, Dp[p].dtype)
            Dpl_X = Dp[p][:, :-1]
            Dpl_y = Dp[p][:, -1]
            # Print shapes and types of every X and y arrays
            print("\tDpl_X: ", Dpl_X.shape, Dpl_X.dtype)
            print("\tDpl_y: ", Dpl_y.shape, Dpl_y.dtype)
            print("\tDpl_y class = 0: ", np.count_nonzero(Dpl_y == 0))
            print("\tDpl_y class = 1: ", np.count_nonzero(Dpl_y == 1))

            svm = SVC(**params)
            svm.fit(Dpl_X, Dpl_y)

            alpha = get_lagrangian_multiplier(svm)
            bias = svm.intercept_   # TODO Bias?
            
            sv = svm.support_vectors_
            sv_y = Dpl_y[svm.support_]
            print("\tClass 0 in sv: ", np.count_nonzero(sv_y == 0))
            print("\tClass 1 in sv: ", np.count_nonzero(sv_y == 1))
            nsv = svm.n_support_
            print ("\tsv: ", sv.shape, sv.dtype)
            print ("\tnsv: ", nsv.shape, nsv.dtype, nsv)
            print ("\talpha: ", alpha.shape, alpha.dtype)
            # SV shape: (nsv, 1004)
            # Alpha shape: (1, nsv)
            sv_with_label = np.append(sv, sv_y.reshape(-1, 1), axis=1)
            print("\tSV with label: ", sv_with_label.shape, sv_with_label.dtype)
            print("\tSV with label class 0: ", np.count_nonzero(sv_with_label[:, -1] == 0))
            print("\tSV with label class 1: ", np.count_nonzero(sv_with_label[:, -1] == 1))
            Slp.append(sv[alpha[0, :] > 0])

            if P == 1:
                S = Slp
                return S, alpha, svm, Dpl_X, Dpl_y
            else:
                gamma = svm_score(sv, alpha, svm)
                Tgamma = np.mean(gamma) #TODO Test
                Rpl = get_relevant_vectors(sv_with_label, gamma, Tgamma)
                # TODO Send/Receive
                print("\tRpl len: ", len(Rpl))
                print("\tRpl class 0: ", [int(r[-1]) for r in Rpl].count(0))
                print("\tRpl class 1: ", [int(r[-1]) for r in Rpl].count(1))
                
                next_level_D[p].extend(Rpl)
                next_level_alpha.extend(alpha) # TODO Mirar en el paper
        
        Dp = compress_next_level_D(next_level_D, P)
        alpha = next_level_alpha # TODO Mirar en el paper
        P = int(np.ceil(P/2))
        next_level_D = [[] for p in range(P)]
        next_level_alpha = []
        l = l+1

def compress_next_level_D(next_level_D, P):
    """
    Sets the next level D.
    Now we have P partitions, so we need to compress to P/2 partitions.

    Args:
    next_level_D: next level D
    P: number of partitions

    Returns:
    next_level_D: next level D
    """
    next_level_D_uncompressed = next_level_D
    next_level_P = int(np.ceil(P/2))
    next_level_D_compressed = [[] for p in range(next_level_P)]
    for p in range(next_level_P):
        next_level_D_compressed[p].extend(next_level_D_uncompressed[p])
        if p+next_level_P < P:
            next_level_D_compressed[p].extend(next_level_D_uncompressed[p+next_level_P])
        next_level_D_compressed[p] = np.array(next_level_D_compressed[p])
    return next_level_D_compressed

def svm_score(sv, alpha, svm):
    """
    Calculates the distance of each instance in D from the hyperplane
    defined by the SVM.

    Args:
    D: dataset
    sv: support vectors

    Returns:
    gamma: distance of each instance in D from the hyperplane
    """
    return get_distances_from_boundary(sv, alpha, svm)


def get_distances_from_boundary(sv, alpha, svm):
    """
    Calculates the distance of sv from the hyperplane defined by the SVM.
    https://stackoverflow.com/questions/32074239/sklearn-getting-distance-of-each-point-from-decision-boundary

    Args:
    D: dataset (we need it for the gran matrix)
    sv: support vectors
    alpha: lagrangian multiplier
    svm: SVM

    Returns:
    gamma: distance of x from the hyperplane
    """
    # Get svm.decision_function for every instance in D
    y = svm.decision_function(sv)
    
    w_norm = 0
    for i in range(sv.shape[0]):
        for j in range(sv.shape[0]):
            w_norm += alpha[0][i] * alpha[0][j] * rbf_kernel(sv[i], sv[j])
    distances = y / w_norm
    # Get the distance of x from the hyperplane
    gamma = np.abs(distances)
    print("\tGamma: ", "Mean: ", np.mean(gamma), "Std: ", np.std(gamma), "Max: ", np.max(gamma), "Min: ", np.min(gamma))
    return gamma


def get_relevant_vectors(sv, gamma, Tgamma):
    """
    Returns the relevant vectors of Slp.

    Args:
    sv: support vectors
    gamma: distance of each instance in D from the hyperplane
    Tgamma: threshold gamma

    Returns:
    Rpl: relevant vectors of Slp
    """         
    indexes_above_threshold = np.where(gamma >= Tgamma)[0]
    print("\tIndexes above threshold len: ", len(indexes_above_threshold))
    Rpl = [sv[i] for i in indexes_above_threshold]
    return Rpl

def get_lagrangian_multiplier(svm):
    alpha = np.abs(svm.dual_coef_)
    return alpha


def rbf_kernel(x1, x2):
    """
    Computes the RBF kernel between vectors x1 and x2.

    Args:
    x1: first vector
    x2: second vector

    Returns:
    K: RBF kernel between x1 and x2
    """
    sigma = 0.5
    K = np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))
    return K

def group_classwise(D, C):
    """
    Groups D instances by class.

    Args:
    D: dataset
    C: number of classes

    Returns:
    Dc: dataset grouped by class
    """
    Dc = []
    for c in range(C):
        Dc.append(D[D[:, -1] == c])
    return Dc

def group_classwise_by_indexes(D, C):
    """
    Returns the indexes of D instances pertaining to each class.

    Args:
    D: dataset
    C: number of classes

    Returns:
    Dc: indexes of instances pertaining to each class
    """
    Dc = []
    for c in range(C):
        Dc.append(np.where(D[:, -1] == c)[0])
    return Dc

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


# Set run id
if len(sys.argv) < 2:
    run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
else:
    run_id = sys.argv[1]

# Time
start = time.time()

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

# Undersample to keep only 10% of negative instances
X_train_seqs_neg = X_train_seqs_neg[:int(len(X_train_seqs_neg) * 0.1)]

X_train_seqs = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])

X_test_seqs = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])

X_train_seqs, y_train_seqs = shuffle(X_train_seqs, y_train)
X_test_seqs, y_test = shuffle(X_test_seqs, y_test)

# Get a random P% subset of X_train and y_train
subset_train_size = int(sys.argv[2])/100
subset_test_size = int(sys.argv[3])/100

random.seed(42)
train_size = X_train_seqs.shape[0]
idx = random.choice(train_size, int(train_size*subset_train_size), replace=False)
X_train_seqs = X_train_seqs[idx]
y_train = y_train[idx]

test_size = X_test_seqs.shape[0]
idx = random.choice(test_size, int(test_size*subset_test_size), replace=False)
X_test_seqs = X_test_seqs[idx]
y_test = y_test[idx]


# Get one-hot encoding of X_train and X_test
print("Getting one-hot encoding of X_train and X_test...")
encoding_start = time.time()
X_train_onehot = seqs_to_onehot_array(X_train_seqs)
X_test_onehot = seqs_to_onehot_array(X_test_seqs)
encoding_end = time.time()
# Print time elapsed in hours, minutes and seconds
print("Time elapsed: {}h {}m {}s".format(int((encoding_end - encoding_start)/3600), int(((encoding_end - encoding_start)%3600)/60), int((encoding_end - encoding_start)%60)))


# Flatten X_train and X_test
X_train_onehot_flat = X_train_onehot.reshape(X_train_onehot.shape[0], -1)
X_test_onehot_flat = X_test_onehot.reshape(X_test_onehot.shape[0], -1)

# Test
print("X_train_seqs: ", X_train_seqs.shape, X_train_seqs.dtype)
print("X_train_onehot: ", X_train_onehot.shape, X_train_onehot.dtype)
print("X_train_onehot_flat: ", X_train_onehot_flat.shape, X_train_onehot_flat.dtype)
print("y_train_seqs: ", y_train_seqs.shape, y_train_seqs.dtype)

# dataset
D_onehot = np.concatenate((X_train_onehot_flat, y_train.reshape(-1, 1)), axis=1)

print("D_onehot: ", D_onehot.shape, D_onehot.dtype)

# N: number of instance (vectors) in D
N = X_train_onehot_flat.shape[0]
# C: number of classes
C = np.unique(y_train).shape[0]
# P: number of partitions
P = 10
# K: number of clusters
K = 10
# Threshold
Tgamma = 0.9 # TODO Tune this parameter

print("Initial dataset shape: ", X_train_onehot_flat.shape)
print("Initial dataset labels shape: ", y_train.shape)
print("Number of instances (N): ", N)
print("Number of classes (C): ", C)
print("Number of partitions (P): ", P)
print("Number of clusters (K): ", K)

print("X_train_seqs shape: ", X_train_seqs.shape)
print("y_train shape: ", y_train.shape)

# Model
fold_size=X_train_seqs.shape[0]/10
print("Fold size: "+str(fold_size))
params = {'kernel': 'rbf', 'probability': True}
sv, alpha, clf, X_train, y_train = dip_svm_learning(D_onehot, N, P, K, Tgamma, params)
sv = np.array(sv)[0]
print("Support vectors shape", sv.shape)
print("Support vectors", sv)
# Prediction
y_pred_test = clf.predict(X_test_onehot_flat)
y_pred_train = clf.predict(X_train_onehot_flat)
y_proba_test = clf.predict_proba(X_test_onehot_flat)[:, 1]
y_proba_train = clf.predict_proba(X_train_onehot_flat)[:, 1]

# Save results
log_file = "logs/"+run_id+".log"
plot_file = "logs/"+run_id+".png"
model_file = "logs/"+run_id+".pkl"
y_train_file = "logs/"+run_id+"_y_train.pkl"
y_pred_train_file = "logs/"+run_id+"_y_pred_train.pkl"
y_test_file = "logs/"+run_id+"_y_test.pkl"
y_pred_test_file = "logs/"+run_id+"_y_pred_test.pkl"
y_proba_train_file = "logs/"+run_id+"_y_proba_train.pkl"
y_proba_test_file = "logs/"+run_id+"_y_proba_test.pkl"

pickle.dump(clf, open(model_file, 'wb'), protocol=4)
pickle.dump(y_train, open(y_train_file, 'wb'), protocol=4)
pickle.dump(y_test, open(y_test_file, 'wb'), protocol=4)
pickle.dump(y_pred_train, open(y_pred_train_file, 'wb'), protocol=4)
pickle.dump(y_pred_test, open(y_pred_test_file, 'wb'), protocol=4)
pickle.dump(y_proba_train, open(y_proba_train_file, 'wb'), protocol=4)
pickle.dump(y_proba_test, open(y_proba_test_file, 'wb'), protocol=4)

with open(log_file, 'w') as f:
    with redirect_stdout(f):
        print(classification_report(y_test, y_pred_test))

        print('Class 0 y_train: ', np.sum(y_train == 0))
        print('Class 1 y_train: ', np.sum(y_train == 1))

        print('Class 0 y_test: ', np.sum(y_test == 0))
        print('Class 1 y_test: ', np.sum(y_test == 1))
        
        print('Train results:')
        print('\tAccuracy score:', accuracy_score(y_train, y_pred_train))
        print('\tBinary crossentropy:', log_loss(y_train, y_pred_train))
        print('\tAUC ROC:', roc_auc_score(y_train, clf.decision_function(X_train_onehot_flat)))
        print('\tF1 score:', f1_score(y_train, y_pred_train))

        print('Test results:')
        print('\tAccuracy score:', accuracy_score(y_test, y_pred_test))
        print('\tBinary crossentropy:', log_loss(y_test, y_pred_test))
        print('\tAUC ROC:', roc_auc_score(y_test, clf.decision_function(X_test_onehot_flat)))
        print('\tF1 score:', f1_score(y_test, y_pred_test))

        # https://scikit-learn.org/stable/modules/svm.html
        print('Number of support vectors for each class:', clf.support_.shape[0])

        # Time
        print('Elapsed time:', str(timedelta(seconds=time.time() - start)))