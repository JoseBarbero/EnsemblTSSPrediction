import sys
sys.path.append("../utils")
import numpy as np
import re
import os
import pickle
from Results import test_results, plot_train_history
from datetime import datetime
from contextlib import redirect_stdout
import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention

def titer():
    model = Sequential()

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(203, 4)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(LSTM(256, return_sequences=True, go_backwards=False))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    X_train_file = open('../data/onehot_serialized/X_train_TIS.pkl', 'rb')
    y_train_file = open('../data/onehot_serialized/y_train_TIS.pkl', 'rb')
    X_val_file = open('../data/onehot_serialized/X_val_TIS.pkl', 'rb')
    y_val_file = open('../data/onehot_serialized/y_val_TIS.pkl', 'rb')
    X_test_file = open('../data/onehot_serialized/X_test_TIS.pkl', 'rb')
    y_test_file = open('../data/onehot_serialized/y_test_TIS.pkl', 'rb')

    X_train = pickle.load(X_train_file)
    y_train = pickle.load(y_train_file)
    X_val = pickle.load(X_val_file)
    y_val = pickle.load(y_val_file)
    X_test = pickle.load(X_test_file)
    y_test = pickle.load(y_test_file)

    X_train_file.close()
    y_train_file.close()
    X_val_file.close()
    y_val_file.close()
    X_test_file.close()
    y_test_file.close()

    y_train = y_train.reshape(*y_train.shape, 1)
    y_val = y_val.reshape(*y_val.shape, 1)
    y_test = y_test.reshape(*y_test.shape, 1)


    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    else:
        run_id = sys.argv[1]
        #run_id = "".join(categories)

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"

    model = titer()
    model.build(X_train.shape)
    model.summary()
    
    logdir = os.path.dirname(log_file)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='max')

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])
            print("Train results:\n")
            test_results(X_train, y_train, model)
            print("Val results:\n")
            test_results(X_val, y_val, model)
            print("Test results:\n")
            test_results(X_test, y_test, model)
            

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)