import os.path
import itertools

from matplotlib import pyplot as plt
import numpy as np


def plot_g_d_loss(g_loss_set, d_loss_set):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 6), dpi=400)
    plt.rc('font', family='Times New Roman')
    plt.plot(g_loss_set, 'b', label="Generator", linewidth=2.5)
    plt.plot(d_loss_set, 'r', label="Discriminator", linewidth=2.5)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_accuracy(train_acc_set, val_acc_set):
    plt.figure(figsize=(8, 6))
    plt.rc('font', family='Times New Roman')
    epochs = len(train_acc_set)
    x = range(1, epochs + 1)
    plt.plot(x, train_acc_set, 'r', label='train-acc')
    plt.plot(x, val_acc_set, 'b', label='val-acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def plot_loss(train_loss_set, val_loss_set):
    plt.figure(figsize=(8, 6))
    epochs = len(train_loss_set)
    x = range(1, epochs + 1)
    plt.plot(x, train_loss_set, 'r', label='train-loss')
    plt.plot(x, val_loss_set, 'b', label='val-loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_confusion_matrix(cm, save_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    classes = ('normal', 'abnormal')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    plt.rc('font', family='Times New Roman')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20, rotation=90)

    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=28)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cm.jpg'), bbox_inches='tight')
    plt.show()
