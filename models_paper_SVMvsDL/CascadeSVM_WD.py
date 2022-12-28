import sys
import os
sys.path.append("../utils")
from WDKernel import wdkernel_gram_matrix, get_K_value, parallel_wdkernel_gram_matrix
import time
from datetime import timedelta
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import classification_report  # classfication summary
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pickle 


'''
Original code from: https://github.com/fhebert/CascadeSVC
'''

class CascadeSVM_WD():
    def __init__(self, fold_size=10000, verbose=True, C=1, kernel="precomputed", degree=3,
                 gamma="scale", coef0=0.0, probability=True):
        self.fold_size = fold_size
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
        if X.shape[0]<2*self.fold_size:
            print("The number of instances is lower than 2*fold_size")
            print("A simple SVC estimation is performed")
            print("The following estimator is used: "+str(self.base_svc))
            id, X, y = self.__get_sv__(np.arange(X.shape[0]), X, y)
            self.n_steps = 1
            self.support_ = id
        else:
            n_init = X.shape[0]
            k = 1
            if self.verbose:
                print("Cascade step "+str(k))
                print("Total number of instances: "+str(n_init))
            id = np.arange(X.shape[0]).astype("int")
            id, X, y = self.__kfold_svc__(id, X, y)
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
            id, X, y = self.__get_sv__(id, X, y)
            self.nsteps = k
            self.support_ = id
        if self.verbose:
            print("Final number of support vectors: "+str(len(self.support_)))
        return n_new
    
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

X_train = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])

X_test = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])

# Get a random 1% subset of X_train and y_train
subset_pc_size = int(sys.argv[2])/100

random.seed(42)
train_size = X_train.shape[0]
idx = random.choice(train_size, int(train_size*subset_pc_size), replace=False)
X_train = X_train[idx]
y_train = y_train[idx]

test_size = X_test.shape[0]
idx = random.choice(test_size, int(test_size*subset_pc_size), replace=False)
X_test = X_test[idx]
y_test = y_test[idx]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Model
fold_size=X_train.shape[0]/10
print("Fold size: "+str(fold_size))
clf = CascadeSVM_WD(fold_size=fold_size,C=0.1,gamma=0.1,kernel="precomputed", probability=True)
X_train = clf.fit(X_train, y_train)

# Prediction
X_test_gram = parallel_wdkernel_gram_matrix(X_test, X_train)
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
        print('Number of support vectors for each class:', clf.n_support_)

        # Time
        print('Elapsed time:', str(timedelta(seconds=time.time() - start)))