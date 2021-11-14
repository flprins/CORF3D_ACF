from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def train_test_split(n_split):

    """

    Function to return train and test sets

    :param n_split : number of splits
    :return: Object for splitting dataset in a stratified way

    """

    skf = StratifiedShuffleSplit(n_splits=n_split, random_state=None, test_size=0.2)

    return skf


def train_model(model, X_train, y_train, batch_size, num_epochs, X_test, y_test, model_name,
                counter):
    """

    Function to Train model

    :param counter: Number of folds
    :param X_test: Testing data
    :param X_train: Training data
    :param y_train: Training labels
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :param y_test: Testing labels
    :param model_name: Name of model
    :param model: Model to be trained
    :return: Trained model, History of training, Loss, Accuracy

    """

    filepath = "./models/" + str(model_name) + "_" + str(counter) + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                     verbose=1,
                     validation_data=(X_test, y_test), callbacks=callbacks_list)

    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    return model, hist, loss, accuracy


def five_cross_validation(dataset, labels, skf):

    """

      Function to return indexs of five cross validation

      :param dataset: Dataset used for training and testing
      :param labels: Dataset labels
      :param skf: Corss validation split

      :return: indexs of training and testing data

    """

    X_train_index = []
    X_test_index = []
    y_train_index = []
    y_test_index = []

    for train_index, test_index in skf.split(dataset, labels):
        X_train_index.append(train_index)
        X_test_index.append(test_index)
        y_train_index.append(train_index)
        y_test_index.append(test_index)

    return X_train_index, X_test_index, y_train_index, y_test_index


def leave_one_day_out(timestamps, dataset, labels):

    """

      Function to return index of leave one day out method

      :param timestamps: timestamps of the captured images
      :param dataset: Dataset used for training and testing
      :param labels: Dataset labels

      :return: index of training and testing data

    """

    df = pd.read_excel(timestamps, engine='openpyxl')

    data_list = []
    data_list.extend(df['Day no'].tolist())

    X_train_index = []
    X_test_index = []
    y_train_index = []
    y_test_index = []
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=data_list)

    for train_index, test_index in logo.split(dataset, labels, data_list):
        X_train_index.append(train_index)
        X_test_index.append(test_index)
        y_train_index.append(train_index)
        y_test_index.append(test_index)

    return X_train_index, X_test_index, y_train_index, y_test_index

