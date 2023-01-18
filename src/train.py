import os.path

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut, StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def train_test_split(n_split, random_state=None):

    """

    Function to return train and test sets

    :param n_split: number of splits
    :param random_state: rng state for debugging
    :return: Object for splitting dataset in a stratified way

    """

    # skf = StratifiedShuffleSplit(n_splits=n_split, random_state=random_state, test_size=0.2)
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)

    return skf


def train_model(model, X_train, y_train, batch_size, num_epochs, X_test, y_test, model_name,
                counter, feature_map):
    """

    Function to Train model

    :param feature_map: Name of the feature map on which the model is trained
    :param counter: Number of folds
    :param X_test: Testing data
    :param X_train: Training data
    :param y_train: Training labels
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :param y_test: Testing labels
    :param model_name: Name of model
    :param model: Model to be trained
    :return: Trained model, History of training, Loss, Accuracy, Predictions

    """

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    filepath = os.path.join(model_path, f'{model_name}_{counter}_{feature_map}_model.h5')
    earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', patience=7, verbose=2,
                                  mode='auto')
    checkpoint = ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True,
                                 monitor='val_categorical_accuracy', mode='auto')
    callbacks_list = [earlyStopping, checkpoint]

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                     verbose=2, validation_data=(X_test, y_test), callbacks=callbacks_list)

    model.load_weights(filepath)

    y_pred = model.predict(X_test)

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    print(f'[INFO] loss={loss:.4f}, accuracy: {accuracy*100:.4f}%')

    return model, hist, loss, accuracy, y_pred


def five_cross_validation(dataset, labels):
    """
    Function to return indices of five cross validation

    :param dataset: Dataset used for training and testing
    :param labels: Dataset labels

    :return: indices of training and testing data
    """

    train_index = []
    test_index = []

    skf = train_test_split(5)

    for train_idx, test_idx in skf.split(dataset, labels.argmax(1)):
        train_index.append(train_idx)
        test_index.append(test_idx)

    return train_index, test_index


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

    train_index = []
    test_index = []
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups=data_list)

    for train_idx, test_idx in logo.split(dataset, labels, data_list):
        train_index.append(train_idx)
        test_index.append(test_idx)

    return train_index, test_index
