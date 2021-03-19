import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from math import pi
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, fowlkes_mallows_score, cohen_kappa_score, precision_score, recall_score
from ReadData import read_data_as_img, read_data_st
from tensorflow.keras.models import load_model

def report_results_imagedata(model_name, model_path, test_data_dir):
    df_list = []
    for temp in range(310, 365, 5):
        row_dict = {}
        
        X_test, y_test = read_data_as_img(test_data_dir, f"OPNat{temp}K.hg17-test")
        
        model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        y_test_pred = model.predict(X_test).round().astype(int)

        row_dict["temperature"] = temp
        row_dict["f1score"] = f1_score(y_test, y_test_pred)
        row_dict["accuracy"] = accuracy_score(y_test, y_test_pred)
        row_dict["AUC"] = roc_auc_score(y_test, y_test_pred)
        y_test_pred = y_test_pred.reshape(y_test_pred.shape[0],)
        row_dict["gmean"] = fowlkes_mallows_score(y_test, y_test_pred)
        row_dict["kappa"] = cohen_kappa_score(y_test, y_test_pred)
        row_dict["precision"] = precision_score(y_test, y_test_pred)
        row_dict["recall"] = recall_score(y_test, y_test_pred)
        df_list.append(row_dict)
    results_df = pd.DataFrame(df_list)
    return results_df


def report_results_st(model_name, model_path, test_data_dir):
    df_list = []
    row_dict = {}
    
    X_test, y_test = read_data_st(test_data_dir, "test")
    
    model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
    y_test_pred = model.predict(X_test).round().astype(int)

    row_dict["f1score"] = f1_score(y_test, y_test_pred)
    row_dict["accuracy"] = accuracy_score(y_test, y_test_pred)
    row_dict["AUC"] = roc_auc_score(y_test, y_test_pred)
    y_test_pred = y_test_pred.reshape(y_test_pred.shape[0],)
    row_dict["gmean"] = fowlkes_mallows_score(y_test, y_test_pred)
    row_dict["kappa"] = cohen_kappa_score(y_test, y_test_pred)
    row_dict["precision"] = precision_score(y_test, y_test_pred)
    row_dict["recall"] = recall_score(y_test, y_test_pred)
    df_list.append(row_dict)
    results_df = pd.DataFrame(df_list)
    return results_df


def make_spider_by_temp(models_dfs):
    plt.figure(figsize=(15, 25))

    # Create a color palette:
    palette = plt.cm.get_cmap("autumn_r", len(models_dfs[0][0].index))

    # Loop to plot
    row = 0
    for model_df, model_name in models_dfs:
        # number of variable
        categories=list(model_df)[1:]
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(3, 2, row+1, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ["0.2", "0.4", "0.6", "0.8", "1"], color="grey", size=7)
        plt.ylim(0,1)

        idx = 0
        for temp in range(310, 365, 5):
            values=model_df.loc[idx].drop('temperature').values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', color=palette(idx), label=temp)
            #ax.fill(angles, values, 'b', alpha=0.1)
            idx += 1
        row += 1
        # Add a title
        plt.title(model_name.upper(), size=11, color="white", y=1.1)

    plt.figlegend(model_df["temperature"], loc = 'upper left')


def plot_train_history(history, imgname):    
    
    fig, axs = plt.subplots(3)
    
    fig.set_size_inches(15, 20)
    
    fig.suptitle('Model train history', fontsize=22)
    
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('model accuracy')
    axs[0].set(xlabel='epoch', ylabel='accuracy')
    axs[0].legend(['train', 'val'], loc='best')
    axs[0].set_ylim([0, 1])
    
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set(xlabel='epoch', ylabel='loss')
    axs[1].legend(['train', 'val'], loc='best')
    
    axs[2].plot(history['auc'])
    axs[2].plot(history['val_auc'])
    axs[2].set_title('model auc')
    axs[2].set(xlabel='epoch', ylabel='auc')
    axs[2].legend(['train', 'val'], loc='best')
    axs[2].set_ylim([0, 1])
    
    fig.savefig(imgname)
    plt.close()
    

def test_results(X_test, y_test, model):
    test_bc, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=False)

    print(f"\tAccuracy score:  {test_acc}")
    print()
    print(f"\tBinary crossentropy : {test_bc}")
    print()
    print(f"\tAUC ROC: {test_auc}")
    print()
