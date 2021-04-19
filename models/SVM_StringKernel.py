import sys
sys.path.append("../utils")
from ReadData import seqfile_to_instances
from WDKernel import wdkernel, get_K_value
import time

from strkernel.mismatch_kernel import MismatchKernel
from strkernel.mismatch_kernel import preprocess

from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report  # classfication summary
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

import pickle 

X_train_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_train_TISseqs_pos.txt')[::10]
print('xtrp', len(X_train_seqs_pos))
X_train_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_train_TISseqs_neg.txt')[::100]
print('xtrn', len(X_train_seqs_neg))
#X_val_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_val_TISseqs_pos.txt')
#X_val_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_val_TISseqs_neg.txt')
X_test_seqs_pos = seqfile_to_instances('../data/TIS/seqs/X_test_TISseqs_pos.txt')
X_test_seqs_neg = seqfile_to_instances('../data/TIS/seqs/X_test_TISseqs_neg.txt')


# train
start = time.process_time()
X_train_seqs_pos_Kernels = wdkernel(X_train_seqs_pos, d=3)
print('Pos train vectorized', time.process_time() - start)

start = time.process_time()
X_train_seqs_neg_Kernels = wdkernel(X_train_seqs_neg, d=3)
print('Neg train vectorized', time.process_time() - start)

# merge data
# label positive data as 1, negative as 0
Y_train_seqs_pos = np.ones(len(X_train_seqs_pos), dtype=int)
Y_train_seqs_neg = np.zeros(len(X_train_seqs_neg), dtype=int)

print('Y_train_seqs_pos shape', Y_train_seqs_pos.shape)
print('Y_train_seqs_neg shape', Y_train_seqs_neg.shape)

X_train = np.concatenate([X_train_seqs_pos_Kernels, X_train_seqs_neg_Kernels])
y_train = np.concatenate([Y_train_seqs_pos, Y_train_seqs_neg])

# test
start = time.process_time()
X_test_seqs_pos_Kernels = wdkernel(X_test_seqs_pos, d=3)
print('Pos test vectorized', time.process_time() - start)

start = time.process_time()
X_test_seqs_neg_Kernels = wdkernel(X_test_seqs_neg, d=3)
print('Neg test vectorized', time.process_time() - start)

# merge data
# label positive data as 1, negative as 0
Y_test_seqs_pos = np.ones(len(X_test_seqs_pos), dtype=int)
Y_test_seqs_neg = np.zeros(len(X_test_seqs_neg), dtype=int)
X_test = np.concatenate([X_test_seqs_pos_Kernels,X_test_seqs_neg_Kernels])
y_test = np.concatenate([Y_test_seqs_pos,Y_test_seqs_neg])

clf = SVC()
clf.fit(X_train, y_train)


y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

y_score = clf.decision_function(X_test)

# compute true positive rate and false positive rate
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)  # compute auc

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating curve')
plt.legend(loc="lower right")
plt.show()