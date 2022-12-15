import sys
import os
sys.path.append("../utils")
import time
from contextlib import redirect_stdout
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score
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
X_train_file = open('../data/TSS/onehot_serialized/X_train_TSS.pkl', 'rb')
y_train_file = open('../data/TSS/onehot_serialized/y_train_TSS.pkl', 'rb')
X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')

X_train = pickle.load(X_train_file)
y_train = pickle.load(y_train_file)
X_test = pickle.load(X_test_file)
y_test = pickle.load(y_test_file)

X_train_file.close()
X_test_file.close()
y_train_file.close()
y_test_file.close()

subset_pc_size = sys.argv[2]/100

# Get a random 1% subset of X_train and y_train
random.seed(42)
train_size = X_train.shape[0]
idx = random.choice(train_size, int(train_size*subset_pc_size), replace=False)
X_train = X_train[idx]
y_train = y_train[idx]

test_size = X_test.shape[0]
idx = random.choice(test_size, int(test_size*subset_pc_size), replace=False)
X_test = X_test[idx]
y_test = y_test[idx]

# Flatten X_train and X_test
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Model with rbf kernel
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Prediction
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Save results
log_file = "logs/"+run_id+".log"
plot_file = "logs/"+run_id+".png"
model_file = "logs/"+run_id+".pkl"

log_file = "logs/"+run_id+".log"
X_train_file = "logs/"+run_id+"_X_train.pkl"
X_test_file = "logs/"+run_id+"_X_test.pkl"
y_train_file = "logs/"+run_id+"_y_train.pkl"
y_pred_train_file = "logs/"+run_id+"_y_pred_train.pkl"
y_test_file = "logs/"+run_id+"_y_test.pkl"
y_pred_test_file = "logs/"+run_id+"_y_pred_test.pkl"

pickle.dump(clf, open(model_file, 'wb'))
pickle.dump(X_train, open(X_train_file, 'wb'))
pickle.dump(X_test, open(X_test_file, 'wb'))
pickle.dump(y_train, open(y_train_file, 'wb'))
pickle.dump(y_test, open(y_test_file, 'wb'))
pickle.dump(y_pred_train, open(y_pred_train_file, 'wb'))
pickle.dump(y_pred_test, open(y_pred_test_file, 'wb'))

with open(log_file, 'w') as f:
    with redirect_stdout(f):
        print(classification_report(y_test, y_pred_test))
        
        print('Train results:')
        print('\tAccuracy score:', accuracy_score(y_train, y_pred_train))
        print('\tBinary crossentropy:', log_loss(y_train, y_pred_train))
        print('\tAUC ROC:', roc_auc_score(y_train, clf.decision_function(X_train)))
        print('\tF1 score:', f1_score(y_train, y_pred_train))

        print('Test results:')
        print('\tAccuracy score:', accuracy_score(y_test, y_pred_test))
        print('\tBinary crossentropy:', log_loss(y_test, y_pred_test))
        print('\tAUC ROC:', roc_auc_score(y_test, clf.decision_function(X_test)))
        print('\tF1 score:', f1_score(y_test, y_pred_test))

        # https://scikit-learn.org/stable/modules/svm.html
        print('Number of support vectors for each class:', clf.n_support_)

        # Time formatted in days, hours, minutes and seconds
        print(f"Time elapsed: {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")

# Plot results
y_score = clf.decision_function(X_test)
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
