import sys
import os
sys.path.append("../utils")
from WDKernel import wdkernel_gram_matrix, get_K_value, parallel_wdkernel_gram_matrix
import time
from datetime import timedelta
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, f1_score
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
X_train_gram = parallel_wdkernel_gram_matrix(X_train, X_train)
print('X_test shape:', X_test.shape)
X_test_gram = parallel_wdkernel_gram_matrix(X_test, X_train)    # Importante que lo segundo sea la matriz de train https://stackoverflow.com/questions/26962159/how-to-use-a-custom-svm-kernel


# Model
clf = SVC(kernel='precomputed', probability=True)
clf.fit(X_train_gram, y_train)

# Prediction
y_pred_test = clf.predict(X_test_gram)
y_pred_train = clf.predict(X_train_gram)
y_proba_test = clf.predict_proba(X_test_gram)[:, 1]
y_proba_train = clf.predict_proba(X_train_gram)[:, 1]

# Save results
log_file = "logs/"+run_id+".log"
plot_file = "logs/"+run_id+".png"
model_file = "logs/"+run_id+".pkl"
X_train_file = "logs/"+run_id+"_X_train.pkl"
X_test_file = "logs/"+run_id+"_X_test.pkl"
y_train_file = "logs/"+run_id+"_y_train.pkl"
y_pred_train_file = "logs/"+run_id+"_y_pred_train.pkl"
y_test_file = "logs/"+run_id+"_y_test.pkl"
y_pred_test_file = "logs/"+run_id+"_y_pred_test.pkl"
y_proba_train_file = "logs/"+run_id+"_y_proba_train.pkl"
y_proba_test_file = "logs/"+run_id+"_y_proba_test.pkl"

pickle.dump(clf, open(model_file, 'wb'), protocol=4)
pickle.dump(X_train, open(X_train_file, 'wb'), protocol=4)
pickle.dump(X_test, open(X_test_file, 'wb'), protocol=4)
pickle.dump(y_train, open(y_train_file, 'wb'), protocol=4)
pickle.dump(y_test, open(y_test_file, 'wb'), protocol=4)
pickle.dump(y_pred_train, open(y_pred_train_file, 'wb'), protocol=4)
pickle.dump(y_pred_test, open(y_pred_test_file, 'wb'), protocol=4)
pickle.dump(y_proba_train, open(y_proba_train_file, 'wb'), protocol=4)
pickle.dump(y_proba_test, open(y_proba_test_file, 'wb'), protocol=4)

with open(log_file, 'w') as f:
    with redirect_stdout(f):
        print('Class 0 y_train: ', np.sum(y_train == 0))
        print('Class 1 y_train: ', np.sum(y_train == 1))

        print('Class 0 y_test: ', np.sum(y_test == 0))
        print('Class 1 y_test: ', np.sum(y_test == 1))
        
        
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

# Plot results
y_score = clf.decision_function(X_test_gram)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating curve')
# plt.legend(loc="lower right")
# plt.savefig(plot_file)