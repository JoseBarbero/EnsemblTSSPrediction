import sys
sys.path.append("../utils")
import numpy as np
import re
import os
import pickle
import time
from Results import test_results, plot_train_history, recall_m, precision_m, f1_m
from datetime import datetime
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

def cnn_blstm():
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, data_format='channels_last', activation='relu', input_shape=(1003, 4)))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(64, return_sequences=True, go_backwards=False)))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["accuracy", f1_m, 'AUC'])

    return model

def k_train(model_definition, n_folds, global_X_train, global_X_val, global_X_test, global_y_train, global_y_val, global_y_test, run_id):
    
    accuracy_train = np.zeros(5)
    binarycrossentropy_train = np.zeros(5)
    f1_train = np.zeros(5)
    auc_train = np.zeros(5)

    accuracy_val = np.zeros(5)
    binarycrossentropy_val = np.zeros(5)
    f1_val = np.zeros(5)
    auc_val = np.zeros(5)

    accuracy_test = np.zeros(5)
    binarycrossentropy_test = np.zeros(5)
    f1_test = np.zeros(5)
    auc_test = np.zeros(5)

    summary_file = "logs/"+run_id+".log"

    # Shuffle X_train and y_train keeping the same order
    shuffled_X_train_idx = np.random.permutation(len(global_X_train))
    global_X_train = global_X_train[shuffled_X_train_idx]
    global_y_train = global_y_train[shuffled_X_train_idx]

    shuffled_X_val_idx = np.random.permutation(len(global_X_val))
    global_X_val = global_X_val[shuffled_X_val_idx]
    global_y_val = global_y_val[shuffled_X_val_idx]

    # Split data in 5 groups for cross validation
    X_train_splits = np.array_split(global_X_train, n_folds)
    y_train_splits = np.array_split(global_y_train, n_folds)

    X_val_splits = np.array_split(global_X_val, n_folds)
    y_val_splits = np.array_split(global_y_val, n_folds)

    for k in range(n_folds):

        X_train = X_train_splits[k]
        y_train = y_train_splits[k]
        X_val = X_val_splits[k]
        y_val = y_val_splits[k]
        X_test = global_X_test
        y_test = global_y_test

        log_file = "logs/"+run_id+"_"+str(k+1)+"_fold.log"
        hist_file = "logs/"+run_id+"_"+str(k+1)+"_fold.pkl"
        plot_file = "logs/"+run_id+"_"+str(k+1)+"_fold.png"
        model_file = "logs/"+run_id+"_"+str(k+1)+"_fold.h5"
        X_train_file = "logs/"+run_id+"_"+str(k+1)+"_fold_X_train.pkl"
        X_val_file = "logs/"+run_id+"_"+str(k+1)+"_fold_X_val.pkl"
        X_test_file = "logs/"+run_id+"_"+str(k+1)+"_fold_X_test.pkl"
        y_train_file = "logs/"+run_id+"_"+str(k+1)+"_fold_y_train.pkl"
        y_val_file = "logs/"+run_id+"_"+str(k+1)+"_fold_y_val.pkl"
        y_test_file = "logs/"+run_id+"_"+str(k+1)+"_fold_y_test.pkl"
        y_pred_file = "logs/"+run_id+"_"+str(k+1)+"_fold_y_pred.pkl"

        logdir = os.path.dirname(log_file)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        model = model_definition
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
                    pickle.dump(history.history, file_pi, protocol=4)

                with open(X_train_file, 'wb') as file_pi:
                    pickle.dump(X_train, file_pi, protocol=4)
                
                with open(X_val_file, 'wb') as file_pi:
                    pickle.dump(X_val, file_pi, protocol=4)
                
                with open(X_test_file, 'wb') as file_pi:
                    pickle.dump(X_test, file_pi, protocol=4)
                
                with open(y_train_file, 'wb') as file_pi:
                    pickle.dump(y_train, file_pi, protocol=4)
                
                with open(y_val_file, 'wb') as file_pi:
                    pickle.dump(y_val, file_pi, protocol=4)
                
                with open(y_test_file, 'wb') as file_pi:
                    pickle.dump(y_test, file_pi, protocol=4)
                
                with open(y_pred_file, 'wb') as file_pi:
                    pickle.dump(model.predict(X_test), file_pi, protocol=4)

                # plot_train_history(history.history, plot_file)
        
        binarycrossentropy_train[k], accuracy_train[k], f1_train[k], auc_train[k] = model.evaluate(X_train, y_train, verbose=False)
        binarycrossentropy_val[k], accuracy_val[k], f1_train[k], auc_val[k] = model.evaluate(X_val, y_val, verbose=False)
        binarycrossentropy_test[k], accuracy_test[k], f1_train[k], auc_test[k] = model.evaluate(X_test, y_test, verbose=False)

        with open(hist_file, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        model.save(model_file)

        with open(summary_file, 'w+') as summary_f:
            summary_f.write('accuracy_train: ')
            summary_f.write(str(accuracy_train[k]))
            summary_f.write('\n')
            summary_f.write('accuracy_val: ')
            summary_f.write(str(accuracy_val[k]))
            summary_f.write('\n')
            summary_f.write('accuracy_test: ')
            summary_f.write(str(accuracy_test[k]))
            summary_f.write('\n')
            summary_f.write('binarycrossentropy_train: ')
            summary_f.write(str(binarycrossentropy_train[k]))
            summary_f.write('\n')
            summary_f.write('binarycrossentropy_val: ')
            summary_f.write(str(binarycrossentropy_val[k]))
            summary_f.write('\n')
            summary_f.write('binarycrossentropy_test: ')
            summary_f.write(str(binarycrossentropy_test[k]))
            summary_f.write('\n')
            summary_f.write('f1_train: ')
            summary_f.write(str(f1_train[k]))
            summary_f.write('\n')
            summary_f.write('f1_val: ')
            summary_f.write(str(f1_val[k]))
            summary_f.write('\n')
            summary_f.write('f1_test: ')
            summary_f.write(str(f1_test[k]))
            summary_f.write('\n')
            summary_f.write('auc_train: ')
            summary_f.write(str(auc_train[k]))
            summary_f.write('\n')
            summary_f.write('auc_val: ')
            summary_f.write(str(auc_val[k]))
            summary_f.write('\n')
            summary_f.write('auc_test: ')
            summary_f.write(str(auc_test[k]))
            summary_f.write('\n')

            summary_f.write('')

    with open(summary_file, 'w+') as summary_f:
        summary_f.write('Mean accuracy_train: ')
        summary_f.write(str(accuracy_train.mean()))
        summary_f.write('\n')
        summary_f.write('Mean accuracy_val: ')
        summary_f.write(str(accuracy_val.mean()))
        summary_f.write('\n')
        summary_f.write('Mean accuracy_test: ')
        summary_f.write(str(accuracy_test.mean()))
        summary_f.write('\n')
        summary_f.write('Mean binarycrossentropy_train: ')
        summary_f.write(str(binarycrossentropy_train.mean()))
        summary_f.write('\n')
        summary_f.write('Mean binarycrossentropy_val: ')
        summary_f.write(str(binarycrossentropy_val.mean()))
        summary_f.write('\n')
        summary_f.write('Mean binarycrossentropy_test: ')
        summary_f.write(str(binarycrossentropy_test.mean()))
        summary_f.write('\n')
        summary_f.write('Mean f1_train: ')
        summary_f.write(str(f1_train.mean()))
        summary_f.write('\n')
        summary_f.write('Mean f1_val: ')
        summary_f.write(str(f1_val.mean()))
        summary_f.write('\n')
        summary_f.write('Mean f1_test: ')
        summary_f.write(str(f1_test.mean()))
        summary_f.write('\n')
        summary_f.write('Mean auc_train: ')
        summary_f.write(str(auc_train.mean()))
        summary_f.write('\n')
        summary_f.write('Mean auc_val: ')
        summary_f.write(str(auc_val.mean()))
        summary_f.write('\n')
        summary_f.write('Mean auc_test: ')
        summary_f.write(str(auc_test.mean()))
        summary_f.write('\n')

def single_train(model_definition, X_train, X_val, X_test, y_train, y_val, y_test, run_id):

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png" 
    model_file = "logs/"+run_id+".h5"  
    X_train_file = "logs/"+run_id+"_X_train.pkl"
    X_val_file = "logs/"+run_id+"_X_val.pkl"
    X_test_file = "logs/"+run_id+"_X_test.pkl"
    y_train_file = "logs/"+run_id+"_y_train.pkl"
    y_val_file = "logs/"+run_id+"_y_val.pkl"
    y_test_file = "logs/"+run_id+"_y_test.pkl"
    y_pred_file = "logs/"+run_id+"_y_pred.pkl"

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
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_auc', factor=0.25, patience=3, verbose=1, min_delta=1e-4, mode='max')

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])

            print('Class 0 y_train: ', np.sum(y_train == 0))
            print('Class 1 y_train: ', np.sum(y_train == 1))

            print('Class 0 y_test: ', np.sum(y_test == 0))
            print('Class 1 y_test: ', np.sum(y_test == 1))
            
            print("Train results:\n")
            test_results(X_train, y_train, model)
            print("Val results:\n")
            test_results(X_val, y_val, model)
            print("Test results:\n")
            test_results(X_test, y_test, model)

            
    model.save(model_file)
    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    with open(X_train_file, 'wb') as file_pi:
        pickle.dump(X_train, file_pi, protocol=4)

    with open(X_val_file, 'wb') as file_pi:
        pickle.dump(X_val, file_pi, protocol=4)

    with open(X_test_file, 'wb') as file_pi:
        pickle.dump(X_test, file_pi, protocol=4)

    with open(y_train_file, 'wb') as file_pi:
        pickle.dump(y_train, file_pi, protocol=4)

    with open(y_val_file, 'wb') as file_pi:
        pickle.dump(y_val, file_pi, protocol=4)

    with open(y_test_file, 'wb') as file_pi:
        pickle.dump(y_test, file_pi, protocol=4)

    with open(y_pred_file, 'wb') as file_pi:
        pickle.dump(model.predict(X_test), file_pi, protocol=4)




if __name__ == "__main__":
    #seed = 42
    #np.random.seed(seed)
    #tf.random.set_seed(42)

    # Time
    start = time.time()

    # X_train_file = open('../data/TSS/onehot_serialized/X_train_TSS.pkl', 'rb')
    # y_train_file = open('../data/TSS/onehot_serialized/y_train_TSS.pkl', 'rb')
    # X_val_file = open('../data/TSS/onehot_serialized/X_val_TSS.pkl', 'rb')
    # y_val_file = open('../data/TSS/onehot_serialized/y_val_TSS.pkl', 'rb')
    # X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
    # y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')

    X_train_file = open('../data/TSS/onehot_serialized/mouse_X_train_TSS.pkl', 'rb')
    y_train_file = open('../data/TSS/onehot_serialized/mouse_y_train_TSS.pkl', 'rb')
    X_val_file = open('../data/TSS/onehot_serialized/mouse_X_val_TSS.pkl', 'rb')
    y_val_file = open('../data/TSS/onehot_serialized/mouse_y_val_TSS.pkl', 'rb')
    X_test_file = open('../data/TSS/onehot_serialized/mouse_X_test_TSS.pkl', 'rb')
    y_test_file = open('../data/TSS/onehot_serialized/mouse_y_test_TSS.pkl', 'rb')

    X_train = pickle.load(X_train_file)
    #X_train = X_train[::2]
    y_train = pickle.load(y_train_file)
    #y_train = y_train[::2]
    X_val = pickle.load(X_val_file)
    #X_val = X_val[::2]
    y_val = pickle.load(y_val_file)
    #y_val = y_val[::2]
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
    
    single_train(cnn_blstm(), X_train, X_val, X_test, y_train, y_val, y_test, run_id)
    #k_train(cnn_blstm(), 5, X_train, X_val, X_test, y_train, y_val, y_test, run_id)

    # Time formatted in days, hours, minutes and seconds
    print(f"Time elapsed: {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")