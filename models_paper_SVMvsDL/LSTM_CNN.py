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
from keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold

def cnn_lstm():
    model = Sequential()

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(1003, 4)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(LSTM(128, return_sequences=True, go_backwards=False))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model

def k_train(model_definition, n_folds, X_train, X_val, X_test, y_train, y_val, y_test, run_id):

    
    accuracy_train = np.zeros(5)
    binarycrossentropy_train = np.zeros(5)
    auc_train = np.zeros(5)

    accuracy_val = np.zeros(5)
    binarycrossentropy_val = np.zeros(5)
    auc_val = np.zeros(5)

    accuracy_test = np.zeros(5)
    binarycrossentropy_test = np.zeros(5)
    auc_test = np.zeros(5)

    summary_file = "logs/"+run_id+".log"

    for k in range(n_folds):

        log_file = "logs/"+run_id+str(k)+".log"
        hist_file = "logs/"+run_id+str(k)+".pkl"
        plot_file = "logs/"+run_id+str(k)+".png"

        logdir = os.path.dirname(log_file)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        model = model_definition
        model.build(X_train.shape)
        model.summary()

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
        
        binarycrossentropy_train[k], accuracy_train[k], auc_train[k] = model.evaluate(X_train, y_train, verbose=False)
        binarycrossentropy_val[k], accuracy_val[k], auc_val[k] = model.evaluate(X_val, y_val, verbose=False)
        binarycrossentropy_test[k], accuracy_test[k], auc_test[k] = model.evaluate(X_test, y_test, verbose=False)

        with open(hist_file, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plot_train_history(history.history, plot_file)

    with open(summary_file, 'wb') as summary_f:
        summary_f.write('accuracy_train: ')
        np.savetxt(summary_f, accuracy_train)
        summary_f.write('accuracy_val: ')
        np.savetxt(summary_f, accuracy_val)
        summary_f.write('accuracy_test: ')
        np.savetxt(summary_f, accuracy_test)
        summary_f.write('binarycrossentropy_train: ')
        np.savetxt(summary_f, binarycrossentropy_train)
        summary_f.write('binarycrossentropy_val: ')
        np.savetxt(summary_f, binarycrossentropy_val)
        summary_f.write('binarycrossentropy_test: ')
        np.savetxt(summary_f, binarycrossentropy_test)
        summary_f.write('auc_train: ')
        np.savetxt(summary_f, auc_train)
        summary_f.write('auc_val: ')
        np.savetxt(summary_f, auc_val)
        summary_f.write('auc_test: ')
        np.savetxt(summary_f, auc_test)

        summary_f.write('')

        summary_f.write('Mean accuracy_train: ')
        np.savetxt(summary_f, accuracy_train.mean())
        summary_f.write('Mean accuracy_val: ')
        np.savetxt(summary_f, accuracy_val.mean())
        summary_f.write('Mean accuracy_test: ')
        np.savetxt(summary_f, accuracy_test.mean())
        summary_f.write('Mean binarycrossentropy_train: ')
        np.savetxt(summary_f, binarycrossentropy_train.mean())
        summary_f.write('Mean binarycrossentropy_val: ')
        np.savetxt(summary_f, binarycrossentropy_val.mean())
        summary_f.write('Mean binarycrossentropy_test: ')
        np.savetxt(summary_f, binarycrossentropy_test.mean())
        summary_f.write('Mean auc_train: ')
        np.savetxt(summary_f, auc_train.mean())
        summary_f.write('Mean auc_val: ')
        np.savetxt(summary_f, auc_val.mean())
        summary_f.write('Mean auc_test: ')
        np.savetxt(summary_f, auc_test.mean())


def single_train(model_definition, X_train, X_val, X_test, y_train, y_val, y_test, run_id):

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"  

    logdir = os.path.dirname(log_file)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    model = model_definition
    model.build(X_train.shape)
    model.summary()

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



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    X_train_file = open('../data/TSS/onehot_serialized/X_train_TSS.pkl', 'rb')
    y_train_file = open('../data/TSS/onehot_serialized/y_train_TSS.pkl', 'rb')
    X_val_file = open('../data/TSS/onehot_serialized/X_val_TSS.pkl', 'rb')
    y_val_file = open('../data/TSS/onehot_serialized/y_val_TSS.pkl', 'rb')
    X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
    y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')

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
    
    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    else:
        run_id = sys.argv[1]
    
    single_train(cnn_lstm(), X_train, X_val, X_test, y_train, y_val, y_test, run_id)
