from keras.models import load_model
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

cnn = load_model('logs/100%data/cnn/cnn_100%data_run1.h5')
lstm = load_model('logs/100%data/lstm/lstm_100%data_run1.h5')
bilstm = load_model('logs/100%data/bilstm/bilstm_100%data_run1.h5')
lstmcnn = load_model('logs/100%data/lstmcnn/lstmcnn_100%data_run1.h5')
bilstmcnn = load_model('logs/100%data/bilstmcnn/bilstmcnn_100%data_run1.h5')
#svm = pickle.load(open('logs/svm_50pc.pkl', 'rb'))

X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')
X_test = pickle.load(X_test_file)
y_test = pickle.load(y_test_file)
X_test_file.close()
y_test_file.close()

y_pred_cnn = cnn.predict(X_test).ravel()
y_pred_lstm = lstm.predict(X_test).ravel()
y_pred_bilstm = bilstm.predict(X_test).ravel()
y_pred_lstmcnn = lstmcnn.predict(X_test).ravel()
y_pred_bilstmcnn = bilstmcnn.predict(X_test).ravel()
#y_pred_svm = svm.predict(X_test)

fpr_cnn, tpr_cnn, _ = metrics.roc_curve(y_test,  y_pred_cnn)
fpr_lstm, tpr_lstm, _ = metrics.roc_curve(y_test,  y_pred_lstm)
fpr_bilstm, tpr_bilstm, _ = metrics.roc_curve(y_test,  y_pred_bilstm)
fpr_lstmcnn, tpr_lstmcnn, _ = metrics.roc_curve(y_test,  y_pred_lstmcnn)
fpr_bilstmcnn, tpr_bilstmcnn, _ = metrics.roc_curve(y_test,  y_pred_bilstmcnn)
# fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test,  y_pred_svm)

auc_cnn = metrics.roc_auc_score(y_test, y_pred_cnn)
auc_lstm = metrics.roc_auc_score(y_test, y_pred_lstm)
auc_bilstm = metrics.roc_auc_score(y_test, y_pred_bilstm)
auc_lstmcnn = metrics.roc_auc_score(y_test, y_pred_lstmcnn)
auc_bilstmcnn = metrics.roc_auc_score(y_test, y_pred_bilstmcnn)
#auc_svm = metrics.roc_auc_score(y_test, y_pred_svm)

plt.figure(figsize=(9, 5))

plt.plot(fpr_cnn, tpr_cnn, label="cnn, auc="+str(round(auc_cnn, 3)))
plt.plot(fpr_lstm, tpr_lstm, label="lstm, auc="+str(round(auc_lstm, 3)))
plt.plot(fpr_bilstm, tpr_bilstm, label="bilstm, auc="+str(round(auc_bilstm, 3)))
plt.plot(fpr_lstmcnn, tpr_lstmcnn, label="lstmcnn, auc="+str(round(auc_lstmcnn, 3)))
plt.plot(fpr_bilstmcnn, tpr_bilstmcnn, label="bilstmcnn, auc="+str(round(auc_bilstmcnn, 3)))
#plt.plot(fpr_svm, tpr_svm, label="svm, auc="+str(round(auc_svm, 3)))


plt.legend(loc=4, fontsize=12)
plt.show()
plt.savefig('rocs.png')