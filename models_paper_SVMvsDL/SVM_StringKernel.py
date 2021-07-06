import sys
import os
sys.path.append("../utils")
from ReadData import seqfile_to_instances
from WDKernel import wdkernel_gram_matrix, get_K_value, parallel_wdkernel_gram_matrix
import time
from datetime import timedelta
from contextlib import redirect_stdout
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report  # classfication summary
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pickle 

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
#X_val_seqs_pos_file = open('../data/TSS/seqs/X_val_TSSseqs_pos_chararray.txt', 'rb')
#X_val_seqs_neg_file = open('../data/TSS/seqs/X_val_TSSseqs_neg_chararray.txt', 'rb')
X_test_seqs_pos_file = open('../data/TSS/seqs/X_test_TSSseqs_pos_chararray.txt', 'rb')
X_test_seqs_neg_file = open('../data/TSS/seqs/X_test_TSSseqs_neg_chararray.txt', 'rb')

X_train_seqs_pos = pickle.load(X_train_seqs_pos_file)
X_train_seqs_neg = pickle.load(X_train_seqs_neg_file)
#X_val_seqs_pos = pickle.load(X_val_seqs_pos_file)
#X_val_seqs_neg = pickle.load(X_val_seqs_neg_file)
X_test_seqs_pos = pickle.load(X_test_seqs_pos_file)
X_test_seqs_neg = pickle.load(X_test_seqs_neg_file)

X_train_seqs_pos = X_train_seqs_pos[::2]
X_train_seqs_neg = X_train_seqs_neg[::2]
#X_val_seqs_pos = X_val_seqs_pos[::10]
#X_val_seqs_neg = X_val_seqs_neg[::10]
X_test_seqs_pos = X_test_seqs_pos[::2]
X_test_seqs_neg = X_test_seqs_neg[::2]

X_train_seqs_pos_file.close()
X_train_seqs_neg_file.close()
#X_val_file.close()
#y_val_file.close()
X_test_seqs_pos_file.close()
X_test_seqs_neg_file.close()

# Train
X_train = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])
print('X_train shape:', X_train.shape)
X_train_gram = parallel_wdkernel_gram_matrix(X_train, X_train)


# Test
X_test = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])
print('X_test shape:', X_test.shape)
X_test_gram = parallel_wdkernel_gram_matrix(X_test, X_train)    # Importante que lo segundo sea la matriz de train https://stackoverflow.com/questions/26962159/how-to-use-a-custom-svm-kernel


# Model
clf = SVC(kernel='precomputed')
clf.fit(X_train_gram, y_train)

# Prediction
y_pred_test = clf.predict(X_test_gram)
y_pred_train = clf.predict(X_train_gram)

# Save results
log_file = "logs/"+run_id+".log"
plot_file = "logs/"+run_id+".png"
model_file = "logs/"+run_id+".pkl"

pickle.dump(clf, open(model_file, 'wb'))

with open(log_file, 'w') as f:
    with redirect_stdout(f):
        print(classification_report(y_test, y_pred_test))
        
        print('Train results:')
        print('\tAccuracy score:', accuracy_score(y_train, y_pred_train))
        print('\tBinary crossentropy:', log_loss(y_train, y_pred_train))
        print('\tAUC ROC:', roc_auc_score(y_train, clf.decision_function(X_train_gram)))

        print('Test results:')
        print('\tAccuracy score:', accuracy_score(y_test, y_pred_test))
        print('\tBinary crossentropy:', log_loss(y_test, y_pred_test))
        print('\tAUC ROC:', roc_auc_score(y_test, clf.decision_function(X_test_gram)))

        # https://scikit-learn.org/stable/modules/svm.html
        print('Number of support vectors for each class:', clf.n_support_)

        # Time
        print('Elapsed time:', str(timedelta(seconds=time.time() - start)))

# Plot results
y_score = clf.decision_function(X_test_gram)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating curve')
plt.legend(loc="lower right")
plt.savefig(plot_file)