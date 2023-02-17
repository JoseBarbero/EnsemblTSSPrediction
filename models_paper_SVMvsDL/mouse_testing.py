import sys
import os
import time
sys.path.append("../utils")
from WDKernel import wdkernel_gram_matrix, get_K_value, parallel_wdkernel_gram_matrix
from CascadeSVM_WD import CascadeSVM_WD
from CascadeSVM_RBF import CascadeSVC
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import classification_report  # classfication summary
from sklearn.metrics import log_loss, make_scorer
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pickle 
from Results import test_results, plot_train_history, recall_m, precision_m, f1_m
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from contextlib import redirect_stdout

def f1_score(y_test, y_pred):
   # Get the best threshold to maximize F1 score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Remove NaNs
    f1_scores = f1_scores[~np.isnan(f1_scores)]
    best_f1 = np.max(f1_scores)
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]

    # Get the threshold that maximizes F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_f1

def accuracy_score(y_test, y_pred):
    # Get the best threshold to maximize acc score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    acc_scores = (precision + recall) / 2

    # Remove NaNs
    acc_scores = acc_scores[~np.isnan(acc_scores)]
    best_acc = np.max(acc_scores)

    return best_acc

# ONEHOT
X_train_onehot_file = open('../data/TSS/onehot_serialized/mouse_X_train_TSS.pkl', 'rb')
y_train_onehot_file = open('../data/TSS/onehot_serialized/mouse_y_train_TSS.pkl', 'rb')
X_val_onehot_file = open('../data/TSS/onehot_serialized/mouse_X_val_TSS.pkl', 'rb')
y_val_onehot_file = open('../data/TSS/onehot_serialized/mouse_y_val_TSS.pkl', 'rb')
X_test_onehot_file = open('../data/TSS/onehot_serialized/mouse_X_test_TSS.pkl', 'rb')
y_test_onehot_file = open('../data/TSS/onehot_serialized/mouse_y_test_TSS.pkl', 'rb')

X_train_onehot = pickle.load(X_train_onehot_file)
y_train_onehot = pickle.load(y_train_onehot_file)
X_val_onehot = pickle.load(X_val_onehot_file)
y_val_onehot = pickle.load(y_val_onehot_file)
X_test_onehot = pickle.load(X_test_onehot_file)
y_test_onehot = pickle.load(y_test_onehot_file)

X_train_onehot_file.close()
y_train_onehot_file.close()
X_val_onehot_file.close()
y_val_onehot_file.close()
X_test_onehot_file.close()
y_test_onehot_file.close()

# SEQS
X_train_seqs_pos_file = open('../data/TSS/seqs/mouse_X_train_TSSseqs_pos_chararray.txt', 'rb')
X_train_seqs_neg_file = open('../data/TSS/seqs/mouse_X_train_TSSseqs_neg_chararray.txt', 'rb')
X_test_seqs_pos_file = open('../data/TSS/seqs/mouse_X_test_TSSseqs_pos_chararray.txt', 'rb')
X_test_seqs_neg_file = open('../data/TSS/seqs/mouse_X_test_TSSseqs_neg_chararray.txt', 'rb')

X_train_seqs_pos = pickle.load(X_train_seqs_pos_file)
X_train_seqs_neg = pickle.load(X_train_seqs_neg_file)
X_test_seqs_pos = pickle.load(X_test_seqs_pos_file)
X_test_seqs_neg = pickle.load(X_test_seqs_neg_file)

X_train_seqs_pos_file.close()
X_train_seqs_neg_file.close()
X_test_seqs_pos_file.close()
X_test_seqs_neg_file.close()

X_train_seqs = np.concatenate([X_train_seqs_pos, X_train_seqs_neg])
y_train_seqs = np.concatenate([np.ones(len(X_train_seqs_pos), dtype=int), np.zeros(len(X_train_seqs_neg), dtype=int)])

X_test_seqs = np.concatenate([X_test_seqs_pos, X_test_seqs_neg])
y_test_seqs = np.concatenate([np.ones(len(X_test_seqs_pos), dtype=int), np.zeros(len(X_test_seqs_neg), dtype=int)])

X_test_onehot_flatten = X_test_onehot.reshape(X_test_onehot.shape[0], X_test_onehot.shape[1]*X_test_onehot.shape[2])

cnn_model_path = 'logs/cnn/cnn_1.h5'
lstm_model_path = 'logs/lstm/lstm_1.h5'
lstm_cnn_model_path = 'logs/lstm_cnn/lstm_cnn_1.h5'
bilstm_model_path = 'logs/bilstm/bilstm_1.h5'
bilstm_cnn_model_path = 'logs/bilstm_cnn/bilstm_cnn_1.h5'
cascade_svm_rbf_intelex_model_path = 'logs/cascade_rbf_100pc_intelex/cascade_rbf_100pc_intelex_1.pkl'
cascade_svm_wd_model_path = 'logs/cascade_wd_saving_X.pkl'
svm_wd_X_train_path = 'logs/cascade_wd_saving_X_cascaded_X_train.pkl'

# X_train after cascade for WD
X_train_cascade_wd_file = open(svm_wd_X_train_path, 'rb')
X_train_cascade_wd = pickle.load(X_train_cascade_wd_file)
X_train_cascade_wd_file.close()

# Load models
cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={"f1_m": f1_m})
lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects={"f1_m": f1_m})
lstm_cnn_model = tf.keras.models.load_model(lstm_cnn_model_path, custom_objects={"f1_m": f1_m})
bilstm_model = tf.keras.models.load_model(bilstm_model_path, custom_objects={"f1_m": f1_m})
bilstm_cnn_model = tf.keras.models.load_model(bilstm_cnn_model_path, custom_objects={"f1_m": f1_m})
# svm_wd_model = pickle.load(open(cascade_svm_wd_model_path, 'rb'))
svm_rbf_model = pickle.load(open(cascade_svm_rbf_intelex_model_path, 'rb'))

# Cascade WD Gram matrix
# cascade_wd_gram = parallel_wdkernel_gram_matrix(X_test_seqs, X_train_cascade_wd)

with open("mouse_results.log", 'w') as f:
    with redirect_stdout(f):
        # Predictions
        time_start = time.time()
        print('Predicting...', flush=True)
        cnn_pred = cnn_model.predict(X_test_onehot)
        lstm_pred = lstm_model.predict(X_test_onehot)
        lstm_cnn_pred = lstm_cnn_model.predict(X_test_onehot)
        bilstm_pred = bilstm_model.predict(X_test_onehot)
        bilstm_cnn_pred = bilstm_cnn_model.predict(X_test_onehot)
        # svm_wd_pred = svm_wd_model.predict_proba_from_gram_matrix(cascade_wd_gram)[:, 1]
        svm_rbf_pred = svm_rbf_model.predict_proba(X_test_onehot_flatten)[:, 1]
        print("Prediction finished in {} seconds".format(time.time() - time_start), flush=True)

        # roc_auc_score
        print('Calculating roc_auc_score...', flush=True)
        cnn_roc_auc = roc_auc_score(y_test_onehot, cnn_pred)
        lstm_roc_auc = roc_auc_score(y_test_onehot, lstm_pred)
        lstm_cnn_roc_auc = roc_auc_score(y_test_onehot, lstm_cnn_pred)
        bilstm_roc_auc = roc_auc_score(y_test_onehot, bilstm_pred)
        bilstm_cnn_roc_auc = roc_auc_score(y_test_onehot, bilstm_cnn_pred)
        # svm_wd_roc_auc = roc_auc_score(y_test_seqs, svm_wd_pred)
        svm_rbf_roc_auc = roc_auc_score(y_test_onehot, svm_rbf_pred)

        # f1_score
        print('Calculating f1_score...', flush=True)
        cnn_f1 = f1_score(y_test_onehot, cnn_pred)
        lstm_f1 = f1_score(y_test_onehot, lstm_pred)
        lstm_cnn_f1 = f1_score(y_test_onehot, lstm_cnn_pred)
        bilstm_f1 = f1_score(y_test_onehot, bilstm_pred)
        bilstm_cnn_f1 = f1_score(y_test_onehot, bilstm_cnn_pred)
        # svm_wd_f1 = f1_score(y_test_seqs, svm_wd_pred)
        svm_rbf_f1 = f1_score(y_test_onehot, svm_rbf_pred)

        # Accuracy
        print('Calculating accuracy...', flush=True)
        cnn_acc = accuracy_score(y_test_onehot, cnn_pred)
        lstm_acc = accuracy_score(y_test_onehot, lstm_pred)
        lstm_cnn_acc = accuracy_score(y_test_onehot, lstm_cnn_pred)
        bilstm_acc = accuracy_score(y_test_onehot, bilstm_pred)
        bilstm_cnn_acc = accuracy_score(y_test_onehot, bilstm_cnn_pred)
        # svm_wd_acc = accuracy_score(y_test_seqs, svm_wd_pred)
        svm_rbf_acc = accuracy_score(y_test_onehot, svm_rbf_pred)

        # Print results
        print('Results:')
        print('CNN roc_auc_score: ', cnn_roc_auc)
        print('LSTM roc_auc_score: ', lstm_roc_auc)
        print('LSTM-CNN roc_auc_score: ', lstm_cnn_roc_auc)
        print('BiLSTM roc_auc_score: ', bilstm_roc_auc)
        print('BiLSTM-CNN roc_auc_score: ', bilstm_cnn_roc_auc)
        # print('SVM WD roc_auc_score: ', svm_wd_roc_auc)
        print('SVM RBF roc_auc_score: ', svm_rbf_roc_auc)

        print('CNN f1_score: ', cnn_f1)
        print('LSTM f1_score: ', lstm_f1)
        print('LSTM-CNN f1_score: ', lstm_cnn_f1)
        print('BiLSTM f1_score: ', bilstm_f1)
        print('BiLSTM-CNN f1_score: ', bilstm_cnn_f1)
        # print('SVM WD f1_score: ', svm_wd_f1)
        print('SVM RBF f1_score: ', svm_rbf_f1)

        print('CNN accuracy: ', cnn_acc)
        print('LSTM accuracy: ', lstm_acc)
        print('LSTM-CNN accuracy: ', lstm_cnn_acc)
        print('BiLSTM accuracy: ', bilstm_acc)
        print('BiLSTM-CNN accuracy: ', bilstm_cnn_acc)
        # print('SVM WD accuracy: ', svm_wd_acc)
        print('SVM RBF accuracy: ', svm_rbf_acc)

        print("Finished in {} seconds".format(time.time() - time_start), flush=True)