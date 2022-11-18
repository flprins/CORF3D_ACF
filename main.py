import argparse
import os
import shutil

from keras.models import *
from keras.engine import Model
from sklearn import svm
from sklearn import metrics
import numpy as np

from src.data_load import load_images, binarize_labels, load_corf_arrays
from src.feature_maps import feature_fusion
from src.models import Models
from src.train import five_cross_validation, train_model, leave_one_day_out
from src.visualizations import plot_data_graph


# Code Width Limit #####################################################################################################


def get_args():
    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif v.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', type=str2bool, default=False,
                        help='Whether to use the preprocessing step')
    parser.add_argument('--method', type=str, choices=['5_fold', 'leave_one_day_out'],
                        default="5_fold", help='Select "5_fold", or "leave_one_day_out" method')
    parser.add_argument('--mode', type=str, choices=['single', 'fusion'], default='fusion',
                        help='Select "single", or "fusion" mode')
    parser.add_argument('--dataset_rgb', type=str, default='',
                        required=False,
                        help='Path to RGB dataset')
    parser.add_argument('--dataset_corf', type=str, default='',
                        required=False,
                        help='Path to CORF dataset')
    parser.add_argument('--preprocessed_dataset', type=str, default="./data/preprocessed",
                        required=False,
                        help='Dataset 2 you are using.')
    parser.add_argument('--resize', type=int, default=100,
                        help='Height of cropped input image to network')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of images in each batch')
    parser.add_argument('--classes', type=int, default=383,
                        help='Number of images in each batch')
    parser.add_argument('--trainable', type=str2bool, default=True,
                        help='Whether to train all the layers')
    parser.add_argument('--pretrained_dataset', type=str, default='imagenet',
                        help='specify which pre-trained dataset to use')
    parser.add_argument('--include_top', type=str2bool, default=False,
                        help='specify if to use the top layer of pre-trained model')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--model', type=str, choices=['densenet121', 'mobilenet', 'xception'], default='densenet121',
                        help='Your pre-trained classification model of choice')
    return parser.parse_args()


def single_training(dataset, fold: int, train_index, test_index, binarizelabels, compiled_model, batch_size, num_epochs,
                    model_name, feature_map):
    """
    Plots data graph and returns accuracy and loss for a single model
    """
    X_train, X_test = dataset[train_index[fold]], dataset[test_index[fold]]
    y_train, y_test = binarizelabels[train_index[fold]], binarizelabels[test_index[fold]]

    model, hist, loss, accuracy, y_pred = train_model(compiled_model, X_train, y_train, batch_size, num_epochs, X_test,
                                                      y_test, model_name, fold + 1, feature_map)

    pred_correct = []
    pred_false = []
    # TODO: Give scores (top 3)
    for i in range(len(y_test)):
        if y_test[i].argmax(0) == y_pred[i].argmax(0):
            pred_correct.append(test_index[fold][i])
        else:
            pred_false.append(test_index[fold][i])

    plot_data_graph(hist, num_epochs, fold + 1, model_name, feature_map)
    return accuracy * 100, loss, pred_correct, pred_false


def fusion_training(dataset_1, dataset_2, fold: int, train_index, test_index, binarizelabels, batch_size, num_epochs,
                    model_name, feature_map_1, feature_map_2, trainable, n_classes, pretrained_dataset, include_top,
                    learning_rate):
    """
    Plots data graphs and returns accuracies and losses for two models,
    as well as SVM accuracy for a fusion model
    """
    compiled_model = get_compiled_model(trainable, n_classes, pretrained_dataset, include_top, learning_rate,
                                        model_name)
    fold_accuracy_1, fold_loss_1, pred_correct, pred_false = single_training(dataset_1, fold, train_index, test_index,
                                                                             binarizelabels, compiled_model, batch_size,
                                                                             num_epochs, model_name, feature_map_1)
    compiled_model = get_compiled_model(trainable, n_classes, pretrained_dataset, include_top, learning_rate,
                                        model_name)
    fold_accuracy_2, fold_loss_2, pred_correct, pred_false = single_training(dataset_2, fold, train_index, test_index,
                                                                             binarizelabels, compiled_model, batch_size,
                                                                             num_epochs, model_name, feature_map_2)

    # Convert to single column
    y_train = np.argmax(binarizelabels[train_index[fold]], axis=1)
    y_test = np.argmax(binarizelabels[test_index[fold]], axis=1)

    # Load models
    filepath_1 = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}_{fold+1}_{feature_map_1}_model.h5')
    filepath_2 = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}_{fold+1}_{feature_map_2}_model.h5')

    trained_model_1 = load_model(filepath_1)
    trained_model_2 = load_model(filepath_2)

    # Extract features
    model_1_feature_map = Model(trained_model_1.input,
                                trained_model_1.layers[-2].output)
    model_2_feature_map = Model(trained_model_2.input,
                                trained_model_2.layers[-2].output)

    features_train_1 = model_1_feature_map.predict(dataset_1[train_index[fold]])
    features_test_1 = model_1_feature_map.predict(dataset_1[test_index[fold]])

    features_train_2 = model_2_feature_map.predict(dataset_2[train_index[fold]])
    features_test_2 = model_2_feature_map.predict(dataset_2[test_index[fold]])

    combined_feature_train = feature_fusion(features_train_1, features_train_2)
    combined_feature_test = feature_fusion(features_test_1, features_test_2)

    clf = svm.SVC(kernel='linear')

    clf.fit(combined_feature_train, y_train)

    # Test the model
    y_pred = clf.predict(combined_feature_test)

    svm_accuracy = metrics.accuracy_score(y_test, y_pred)

    return {'acc': fold_accuracy_1, 'loss': fold_loss_1}, {'acc': fold_accuracy_2, 'loss': fold_loss_2}, svm_accuracy


def get_compiled_model(trainable, n_classes, pretrained_dataset, include_top, learning_rate, model_name):
    model = Models(trainable, n_classes, pretrained_dataset, include_top, learning_rate)

    if model_name == 'densenet121':
        model.densenet121()
    elif model_name == 'mobilenet':
        model.mobilenet()
    elif model_name == 'xception':
        model.xception()

    model.model_trainable()
    compiled_model = model.model_compile(learning_rate)
    return compiled_model


def main():
    args = get_args()

    datasets = []

    if args.preprocessing:
        raise NotImplementedError
    else:
        if args.dataset_rgb and args.dataset_corf:
            dataset, labels, labels_list, filenames = load_images(args.dataset_rgb, args.resize)
            datasets.append(dataset)
            dataset, labels, labels_list = load_corf_arrays(args.dataset_corf)
            datasets.append(dataset)
            binarizelabels = binarize_labels(labels)
            model_name = 'RGB'
        elif args.dataset_rgb:
            dataset, labels, labels_list, filenames = load_images(args.dataset_rgb, args.resize)
            datasets.append(dataset)
            binarizelabels = binarize_labels(labels)
            model_name = 'RGB'
        elif args.dataset_corf:
            dataset, labels, labels_list = load_corf_arrays(args.dataset_corf)
            datasets.append(dataset)
            binarizelabels = binarize_labels(labels)
            model_name = 'CORF'
        else:
            raise ValueError('A dataset_rgb or dataset_corf argument is required.')

    all_fold_accuracy = []
    all_fold_loss = []

    if args.method == '5_fold':
        train_index, test_index = five_cross_validation(datasets[0], binarizelabels)
    else:  # method == 'leave-one-out'
        # TODO: Refactor leave_one_day_out to leave_one_phone_out
        train_index, test_index = leave_one_day_out(labels, datasets[0], binarizelabels)

    pred_correct_path = os.path.join(os.path.dirname(__file__), "pred_correct")
    pred_incorrect_path = os.path.join(os.path.dirname(__file__), "pred_incorrect")
    if not os.path.exists(pred_correct_path):
        os.mkdir(pred_correct_path)
    if not os.path.exists(pred_incorrect_path):
        os.mkdir(pred_incorrect_path)

    if args.mode == 'single':
        for fold in range(len(train_index)):
            print(f'[INFO] Starting fold {fold+1}/{len(train_index)}')
            compiled_model = get_compiled_model(args.trainable, len(labels_list), args.pretrained_dataset,
                                                args.include_top, args.learning_rate, args.model)
            packed_tuple = single_training(datasets[0], fold, train_index, test_index, binarizelabels, compiled_model,
                                           args.batch_size, args.num_epochs, args.model, model_name)
            fold_accuracy, fold_loss, pred_correct, pred_false = packed_tuple

            for i in pred_correct:
                shutil.copy2(os.path.join(os.path.dirname(__file__), "data", "Tripod", filenames[i][:-3] + "jpg"),
                             pred_correct_path)
            for i in pred_false:
                shutil.copy2(os.path.join(os.path.dirname(__file__), "data", "Tripod", filenames[i][:-3] + "jpg"),
                             pred_incorrect_path)

            all_fold_accuracy.append(fold_accuracy)
            all_fold_loss.append(fold_loss)

        print('Average accuracy of Model', np.mean(all_fold_accuracy))
        print('Std accuracy of Model', np.std(all_fold_accuracy))

    else:  # mode == fusion
        all_fold_accuracy_2 = []
        all_fold_loss_2 = []
        all_fold_svm_accuracy = []
        for fold in range(len(train_index)):
            results_1, results_2, svm_accuracy = fusion_training(datasets[0], datasets[1], fold, train_index,
                                                                 test_index, binarizelabels, args.batch_size,
                                                                 args.num_epochs, args.model, args.feature_map_1,
                                                                 args.feature_map_2, args.trainable, len(labels_list),
                                                                 args.pretrained_dataset, args.include_top,
                                                                 args.learning_rate)

            all_fold_accuracy.append(results_1['acc'])
            all_fold_loss.append(results_1['loss'])
            all_fold_accuracy_2.append(results_2['acc'])
            all_fold_loss_2.append(results_2['loss'])

            print('Model + SVM accuracy:', svm_accuracy)

            all_fold_svm_accuracy.append(svm_accuracy * 100)

        print('Average accuracy of Model + SVM:', np.mean(all_fold_svm_accuracy))
        print('Std accuracy of Model + SVM:', np.std(all_fold_svm_accuracy))


if __name__ == '__main__':
    main()
