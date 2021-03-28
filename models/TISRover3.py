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
from keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
from keras.callbacks import LearningRateScheduler

def tisrover():
    model = Sequential()

    model.add(Conv2D(filters=50, kernel_size=(9, 4), data_format='channels_last', activation='relu', input_shape=(203, 4, 1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=62, kernel_size=(7, 1), data_format="channels_last", activation='relu'))
    model.add(MaxPooling2D((2,1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=75, kernel_size=(7, 1), strides=1, activation='relu'))
    model.add(MaxPooling2D((2,1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=87, kernel_size=(7, 1), strides=1, activation='relu'))
    model.add(MaxPooling2D((2,1)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=100, kernel_size=(7, 1), strides=1, activation='relu'))
    model.add(MaxPooling2D((2,1)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    
                                                                      
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.05, nesterov=0.9), loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

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

    X_train = X_train.reshape(*X_train.shape, 1)
    X_val = X_val.reshape(*X_val.shape, 1)
    X_test = X_test.reshape(*X_test.shape, 1)

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

    model = tisrover()
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
            
            def scheduler(epoch, lr):
                print(f'Epoch: {epoch}, LR: {lr}')
                if (epoch + 1) % 10 == 0:
                    return lr * 0.75
                else:
                    return lr 
            
            callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=1024,
                                epochs=130,
                                verbose=True,
                                validation_data=(X_val, y_val),
                                callbacks=[callback])

            print("Train results:\n")
            test_results(X_train, y_train, model)
            print("Val results:\n")
            test_results(X_val, y_val, model)
            print("Test results:\n")
            test_results(X_test, y_test, model)
            

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)