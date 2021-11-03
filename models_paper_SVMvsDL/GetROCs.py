from sklearn.metrics import roc_curve
from keras.models import load_model
import pickle

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

y_pred_keras = cnn.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)