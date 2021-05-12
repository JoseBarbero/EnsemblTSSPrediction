import matplotlib.pyplot as plt


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
