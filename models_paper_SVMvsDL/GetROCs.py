from keras.models import load_model
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt

cnn = load_model('logs/100%data/cnn/cnn_100%data_run1.h5')
lstm = load_model('logs/100%data/lstm/lstm_100%data_run1.h5')
bilstm = load_model('logs/100%data/bilstm/bilstm_100%data_run1.h5')
lstmcnn = load_model('logs/100%data/lstmcnn/lstmcnn_100%data_run1.h5')
bilstmcnn = load_model('logs/100%data/bilstmcnn/bilstmcnn_100%data_run1.h5')
svm = pickle.load(open('logs/svm_50pc.pkl', 'rb'))

X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')
X_test = pickle.load(X_test_file)
y_test = pickle.load(y_test_file)
X_test_file.close()
y_test_file.close()

y_pred_cnn = cnn.predict(X_test).ravel()

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_cnn)
auc = metrics.roc_auc_score(y_test, y_pred_cnn)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
plt.savefig('rocs.png')