import sys
import os
sys.path.append("../utils")
import time
from datetime import timedelta
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.svm import SVC
from WDKernel import wdkernel_gram_matrix, get_K_value, parallel_wdkernel_gram_matrix
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

class CascadeDiPSVM_WD():
    def __init__(self, fold_size=10000, folds_idxs=[], verbose=True, C=1, kernel="precomputed", degree=3,
                 gamma="scale", coef0=0.0, probability=True):
        self.fold_size = fold_size
        self.folds_idxs = folds_idxs
        self.verbose = verbose
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        base_svc = SVC(C=self.C, kernel=self.kernel, degree=self.degree,
                       gamma=self.gamma, coef0=self.coef0, probability=self.probability)
        self.base_svc = base_svc
        self._estimator_type = "classifier"

    def __get_sv__(self,id,X,y):
        X_gram_matrix = parallel_wdkernel_gram_matrix(X, X)
        self.base_svc.fit(X_gram_matrix,y)
        ind_sv = self.base_svc.support_
        X = X[ind_sv, :]
        y = y[ind_sv]
        id = id[ind_sv]
        return id, X, y
    
    def __customfold_svc__(self,folds_idxs,X,y):
        Xsv = []
        ysv = []
        idsv = []
        for fold_idxs in folds_idxs:
            a = X[fold_idxs]
            b = y[fold_idxs]
            print("fold_idxs shape: ", fold_idxs.shape)
            print("a shape: ", a.shape)
            print("b shape: ", b.shape)
            id_, X_, y_ = self.__get_sv__(fold_idxs,X[fold_idxs],y[fold_idxs])
            print("X_ shape: ", X_.shape)
            idsv.append(id_)
            Xsv.append(X_)
            ysv.append(y_)
        Xsv = np.vstack(Xsv)
        ysv = np.concatenate(ysv)
        idsv = np.concatenate(idsv)
        return idsv, Xsv, ysv
    
    def __kfold_svc__(self,id,X,y):
        skf = StratifiedKFold(n_splits=round(X.shape[0]/self.fold_size),shuffle=True)
        Xsv = []
        ysv = []
        idsv = []
        for _, ind in skf.split(X,y):
            id_, X_, y_ = self.__get_sv__(id[ind],X[ind,:],y[ind])
            print("X_ shape: ", X_.shape)
            idsv.append(id_)
            Xsv.append(X_)
            ysv.append(y_)
        Xsv = np.vstack(Xsv)
        ysv = np.concatenate(ysv)
        idsv = np.concatenate(idsv)
        return idsv, Xsv, ysv
    
    def fit(self,X,y):
        self.classes_, y = np.unique(y, return_inverse=True)
        n_init = X.shape[0]
        k = 1
        if self.verbose:
            print("Cascade step "+str(k))
            print("Total number of instances: "+str(n_init))
        
        id, X, y = self.__customfold_svc__(self.folds_idxs, X, y)
        n_new = X.shape[0]
        if self.verbose:
            print("Number of remaining instances (support vectors): "+str(n_new))
        while((n_new>2*self.fold_size)&(((n_init-n_new)/n_init)>0.1)):
            k = k+1
            n_init = n_new
            if self.verbose:
                print("Cascade step " + str(k))
            id, X, y = self.__kfold_svc__(id, X, y)
            n_new = X.shape[0]
            if self.verbose:
                print("Number of remaining instances: " + str(n_new))
        k = k+1
        if self.verbose:
            print("Cascade step " + str(k))
        last_train = X
        last_id = id
        id, X, y = self.__get_sv__(id, X, y)
        self.nsteps = k
        self.support_ = id
        if self.verbose:
            print("Final number of support vectors: "+str(len(self.support_)))
        return last_id
    
    def decision_function(self,X):
        return self.base_svc.decision_function(X)
    
    def predict(self,X):
        pred = self.classes_[self.base_svc.predict(X)]
        return pred
    
    def predict_proba(self,X):
        prob = self.base_svc.predict_proba(X)
        return prob
    
    def get_params(self, deep = True):
        res = {"fold_size" : self.fold_size, "verbose" : self.verbose,
               "C" : self.C, "kernel" : self.kernel, "degree" : self.degree,
               "gamma" : self.gamma, "coef0" : self.coef0,
               "probability" : self.probability}
        return res
    
    def set_params(self, **parameters):
        for par, val in parameters.items():
            if par in ["fold_size","verbose"]:
                setattr(self, par, val)
            else:
                setattr(self.base_svc, par, val)
        return self

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
        print("Class: ", c)
        # DcK = dataset of class c clustered into K clusters
        DcK = MiniBatchKMeans(n_clusters=K, random_state=0).fit(Dc[c][:, :-1]) # KMeans causes "core dumped")
        for k in range(K):
            print("\tCluster: ", k)
            # Dck = dataset of class c and cluster k
            Dck = Dc[c][DcK.labels_ == k]
            # Nck = number of instances in class c and cluster k
            Nck = Dck.shape[0]
            for p in range(P):
                print("\t\tPartition: ", p)
                # Nckp = number of instances in class c, cluster k and partition p rounded up
                Nckp = int(np.ceil(Nck/P))
                print("\t\t\tNck: ", Nck)
                print("\t\t\tNckp: ", Nckp)
                U_idxs = random.choice(Nck, Nckp, replace=False)
                # Dckp = dataset of class c, cluster k and partition p
                Dckp = Dck[U_idxs]
                Dp[p].extend(Dckp)
                Dck = np.delete(Dck, U_idxs, axis=0)
                Nck = Dck.shape[0]
    
    return Dp

def distribution_preserving_partitioning_by_indexes(D, N, C, P, K):
    """
    Divides the dataset into pieces that maintain the
    same statistical properties. Returns the indexes of the
    original dataset.

    Args:
    D: dataset
    N: number of instance (vectors) in D
    (d: dimensions) -> deleted, not used
    C: number of classes
    P: number of partitions
    K: number of clusters
    (U: random indexes) -> deleted, used inside the function

    Returns:
    Dp_idxs: partitioned dataset indexes
    """
    Dc_idxs = group_classwise_by_indexes(D, C)
    Dp_idxs = [[] for p in range(P)]
    for c in range(C):
        print("Class: ", c)
        # DcK = dataset of class c clustered into K clusters
        DcKmeans = MiniBatchKMeans(n_clusters=K, random_state=0).fit(D[Dc_idxs[c]][:, :-1]) # KMeans causes "core dumped")
        for k in range(K):
            print("\tCluster: ", k)
            # Dck_idxs = indexes of class c and cluster k
            Dck_idxs = Dc_idxs[c][DcKmeans.labels_ == k]
            # Nck = number of instances in class c and cluster k
            Nck = Dck_idxs.shape[0]
            for p in range(P):
                print("\t\tPartition: ", p)    
                # Nckp = number of instances in class c, cluster k and partition p rounded up
                Nckp = int(np.ceil(Nck/P))
                print("\t\t\tNck: ", Nck)
                print("\t\t\tNckp: ", Nckp)
                # U_idxs Nckp random indexes from Dck_idxs
                U_idxs = random.choice(Nck, Nckp, replace=False)
                # Dckp_idxs = indexes of dataset of class c, cluster k and partition p
                Dckp_idxs = Dck_idxs[U_idxs]
                Dp_idxs[p].extend(Dckp_idxs)
                Dck_idxs = np.delete(Dck_idxs, U_idxs, axis=0)
                Nck = Dck_idxs.shape[0]
    # Return Dp_idxs as a list of arrays
    Dp_idxs = [np.array(Dp_idxs[p]) for p in range(P)]
    return Dp_idxs



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

def dip_svm():
    pass

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

X_train_seqs = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])

X_test_seqs = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])

X_train_seqs, y_train_seqs = shuffle(X_train_seqs, y_train)
X_test_seqs, y_test = shuffle(X_test_seqs, y_test)

# Get a random 1% subset of X_train and y_train
subset_train_size = int(sys.argv[2])/100
subset_test_size = int(sys.argv[3])/100

random.seed(42)
train_size = X_train_seqs.shape[0]
idx = random.choice(train_size, int(train_size*subset_train_size), replace=False)
X_train = X_train_seqs[idx]
y_train = y_train[idx]

test_size = X_test_seqs.shape[0]
idx = random.choice(test_size, int(test_size*subset_test_size), replace=False)
X_test = X_test_seqs[idx]
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
print("X_train_seqs shape: ", X_train_seqs.shape)
print("X_train_onehot shape: ", X_train_onehot.shape)
print("X_train_onehot_flat shape: ", X_train_onehot_flat.shape)
print("y_train_seqs shape: ", y_train_seqs.shape)

# dataset
D = np.concatenate((X_train_onehot_flat, y_train.reshape(-1, 1)), axis=1)
print("D shape: ", D.shape)
# N: number of instance (vectors) in D
N = X_train_onehot_flat.shape[0]
# C: number of classes
C = np.unique(y_train).shape[0]
# P: number of partitions
P = 3 #100
# K: number of clusters
K = 3 #1000

Dp_idxs = distribution_preserving_partitioning_by_indexes(D, N, C, P, K)

print("Initial dataset shape: ", X_train_onehot_flat.shape)
print("Initial dataset labels shape: ", y_train.shape)
print("Number of instances (N): ", N)
print("Number of classes (C): ", C)
print("Number of partitions (P): ", P)
print("Number of clusters (K): ", K)

print("Dp_idxs length: ", len(Dp_idxs))
for i in range(len(Dp_idxs)):
    print("Dp_idxs[", i, "]: ", len(Dp_idxs[i]))

print("X_train_seqs shape: ", X_train_seqs.shape)
print("y_train shape: ", y_train.shape)

# Model
fold_size=X_train_seqs.shape[0]/10
print("Fold size: "+str(fold_size))
clf = CascadeDiPSVM_WD(fold_size=1000,folds_idxs=Dp_idxs,C=0.1,gamma=0.1,kernel="precomputed", probability=True)
train_idx = clf.fit(X_train_seqs, y_train)

# Prediction
X_test_gram = parallel_wdkernel_gram_matrix(X_test_seqs, X_train_seqs[train_idx])
X_train_gram = parallel_wdkernel_gram_matrix(X_train_seqs[train_idx], X_train_seqs[train_idx])
y_train = y_train[train_idx]
y_pred_test = clf.predict(X_test_gram)
y_pred_train = clf.predict(X_train_gram)
y_proba_test = clf.predict_proba(X_test_gram)[:, 1]
y_proba_train = clf.predict_proba(X_train_gram)[:, 1]

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
        
        print('Train results:')
        print('\tAccuracy score:', accuracy_score(y_train, y_pred_train))
        print('\tBinary crossentropy:', log_loss(y_train, y_pred_train))
        print('\tAUC ROC:', roc_auc_score(y_train, clf.decision_function(X_train_gram)))
        print('\tF1 score:', f1_score(y_train, y_pred_train))

        print('Test results:')
        print('\tAccuracy score:', accuracy_score(y_test, y_pred_test))
        print('\tBinary crossentropy:', log_loss(y_test, y_pred_test))
        print('\tAUC ROC:', roc_auc_score(y_test, clf.decision_function(X_test_gram)))
        print('\tF1 score:', f1_score(y_test, y_pred_test))

        # https://scikit-learn.org/stable/modules/svm.html
        print('Number of support vectors for each class:', clf.support_.shape[0])

        # Time
        print('Elapsed time:', str(timedelta(seconds=time.time() - start)))