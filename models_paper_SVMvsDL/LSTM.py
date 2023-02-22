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
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from imblearn.over_sampling import SMOTE

def keep_1to1(X_train, y_train):
    # Reduce negative samples to a 10% of its original size
    X_train_0 = X_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]
    y_train_0 = y_train[y_train == 0]
    y_train_1 = y_train[y_train == 1]

    X_train_0 = X_train_0[:int(X_train_0.shape[0]*0.1)]
    y_train_0 = y_train_0[:int(y_train_0.shape[0]*0.1)]

    X_train = np.concatenate((X_train_0, X_train_1), axis=0)
    y_train = np.concatenate((y_train_0, y_train_1), axis=0)

    return X_train, y_train

def lstm():
    sequence_input = tf.keras.layers.Input(shape=(1003,4))

    x = tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(1003,4))(sequence_input)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid')(output)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", f1_m, 'AUC'])

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

    # Time
    start = time.time()

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
    y_pred_train_file = "logs/"+run_id+"_y_pred_train.pkl"

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

            # Time formatted in days, hours, minutes and seconds
            print(f"Time elapsed: {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")
            

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

    with open(y_pred_train_file, 'wb') as file_pi:
        pickle.dump(model.predict(X_train), file_pi, protocol=4)

    model.save(model_file)


if __name__ == "__main__":
    #seed = 42
    #np.random.seed(seed)
    #tf.random.set_seed(42)

    # Time
    start = time.time()

    # Read data
    species = sys.argv[2]
    if species == "human":
        X_train_file = open('../data/TSS/onehot_serialized/X_train_TSS.pkl', 'rb')
        y_train_file = open('../data/TSS/onehot_serialized/y_train_TSS.pkl', 'rb')
        X_val_file = open('../data/TSS/onehot_serialized/X_val_TSS.pkl', 'rb')
        y_val_file = open('../data/TSS/onehot_serialized/y_val_TSS.pkl', 'rb')
        X_test_file = open('../data/TSS/onehot_serialized/X_test_TSS.pkl', 'rb')
        y_test_file = open('../data/TSS/onehot_serialized/y_test_TSS.pkl', 'rb')
    elif species == "mouse":
        X_train_file = open('../data/TSS/onehot_serialized/mouse_X_train_TSS.pkl', 'rb')
        y_train_file = open('../data/TSS/onehot_serialized/mouse_y_train_TSS.pkl', 'rb')
        X_val_file = open('../data/TSS/onehot_serialized/mouse_X_val_TSS.pkl', 'rb')
        y_val_file = open('../data/TSS/onehot_serialized/mouse_y_val_TSS.pkl', 'rb')
        X_test_file = open('../data/TSS/onehot_serialized/mouse_X_test_TSS.pkl', 'rb')
        y_test_file = open('../data/TSS/onehot_serialized/mouse_y_test_TSS.pkl', 'rb')
    else:
        print("Species not recognized")
        exit()

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
        #run_id = "".join(categories)

    if len(sys.argv) > 3 and sys.argv[3] == "smote":
        # Apply smote to the training set
        smote = SMOTE()
        # Reshape the data to fit the SMOTE algorithm
        original_shape = X_train.shape
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # Reshape the data back to the original shape
        X_train = X_train.reshape(X_train.shape[0], original_shape[1], original_shape[2])
        run_id += "_smote"
    elif len(sys.argv) > 3 and sys.argv[3] == "1to1":
        # Keep only 10% of negative instances
        X_train, y_train = keep_1to1(X_train, y_train)
        run_id += "_1to1"

    single_train(lstm(), X_train, X_val, X_test, y_train, y_val, y_test, run_id)
    #k_train(lstm(), 5, X_train, X_val, X_test, y_train, y_val, y_test, run_id)

    # Time formatted in days, hours, minutes and seconds
    print(f"Time elapsed: {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")