import argparse
import cv2
from sklearn import svm
from sklearn import metrics
from src.data_load import load_images, binarize_labels, load_feature_maps
from src.data_preprocessing import batch_data_preprocessing
from src.evaluation import model_predictions, printWrongPredictions
from src.feature_maps import corf_feature_maps, temp_feature_maps, feature_fusion, \
    rgb_msx_feature_map, feature_stack, normalization
from src.models import Models
from src.train import train_test_split, train_model, five_cross_validation, leave_one_day_out
from src.visualizations import plot_data_graph
# import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', type=str2bool, default=False,
                        help='Whether to use the preprocessing step')
    parser.add_argument('--method', type=str, choices=["5_fold", "leave_one_day_out"],
                        default="5_fold", help='Select "5_fold", or "leave_one_day_out" method')
    parser.add_argument('--mode', type=str, choices=["single", "fusion"],
                        default="fusion",
                        help='Select "single", or "fusion" mode')
    parser.add_argument('--feature_map_1', type=str, choices=["RGB", "MSX"], default="RGB",
                        help='Which Feature map 1 you are using.')
    parser.add_argument('--feature_map_2', type=str, choices=["CORF3D", "TEMP3D", "MSX"],
                        default="", required=False,
                        help='Which Feature map 2 you are using.')
    parser.add_argument('--dataset_1', type=str, default="./data/interim/Pre-segmentation",
                        required=False,
                        help='Dataset 1 you are using.')
    parser.add_argument('--dataset_2', type=str, default="./data/processed/Preprocessed_RGB",
                        required=False,
                        help='Dataset 2 you are using.')
    parser.add_argument('--timestamp', type=str, default="./data/timestamp.xlsx",
                        help='Timestamp file')
    parser.add_argument('--resize', type=int, default=224,
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
    parser.add_argument('--model', type=str, default="densenet121",
                        help='Your pre-trained classification model of choice')
    args = parser.parse_args()

    if args.mode != "fusion" and args.dataset_2 and not args.feature_map_2 == "CORF3D":
        parser.error('Please enter only one dataset')
    if args.mode != "single" and not args.preprocessing:
        parser.error('Please enter both the datasets')
    if args.dataset_1 and not args.preprocessing:
        for f in os.listdir(args.dataset_1):
            name, ext = os.path.splitext(f)
            if ext == '.npy':
                parser.error('Please select folder with RGB or MSX images')

    if args.preprocessing:

        if args.feature_map_1 == "RGB" and args.feature_map_2 == "CORF3D" and args.mode == "fusion":
            print("entered")
            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1)
            corf_feature_set_1 = corf_feature_maps(args.dataset_2, 2.2, 4, 0.0, 0.005)
            corf_feature_set_norm_1 = normalization(corf_feature_set_1)
            corf_feature_set_2 = corf_feature_maps(args.dataset_2, 2.2, 4, 1.8, 0.005)
            corf_feature_set_norm_2 = normalization(corf_feature_set_2)
            corf_feature_set_3 = corf_feature_maps(args.dataset_2, 2.2, 4, 3.6, 0.005)
            corf_feature_set_norm_3 = normalization(corf_feature_set_3)
            dataset_2 = feature_stack(corf_feature_set_norm_1, corf_feature_set_norm_2,
                                      corf_feature_set_norm_3)

        elif args.feature_map_1 == "RGB" and args.feature_map_2 == "TEMP3D" and args.mode == "fusion":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1)
            temp_feature_set_1 = temp_feature_maps(args.dataset_2)
            temp_feature_set_norm_1 = normalization(temp_feature_set_1)
            dataset_2 = feature_stack(temp_feature_set_norm_1, temp_feature_set_norm_1,
                                      temp_feature_set_norm_1)

        elif args.feature_map_1:

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1)

        elif args.feature_map_2 == "TEMP3D" and args.mode == "single":

            temp_feature_set_1 = temp_feature_maps(args.dataset_2)
            temp_feature_set_norm_1 = normalization(temp_feature_set_1)
            dataset_1 = feature_stack(temp_feature_set_norm_1, temp_feature_set_norm_1,
                                             temp_feature_set_norm_1)

        elif args.feature_map_2 == "CORF3D" and args.mode == "single":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1)
            corf_feature_set_1 = corf_feature_maps(args.dataset_2, 2.2, 4, 0, 0.005)
            corf_feature_set_norm_1 = normalization(corf_feature_set_1)
            corf_feature_set_2 = corf_feature_maps(args.dataset_2, 2.2, 4, 1.8, 0.005)
            corf_feature_set_norm_2 = normalization(corf_feature_set_2)
            corf_feature_set_3 = corf_feature_maps(args.dataset_2, 2.2, 4, 3.6, 0.005)
            corf_feature_set_norm_3 = normalization(corf_feature_set_3)
            corf_feature_set = feature_stack(corf_feature_set_norm_1, corf_feature_set_norm_2,
                                             corf_feature_set_norm_3)

        binarizelabels = binarize_labels(labels)

    else:

        if args.feature_map_1 and args.feature_map_2 == "MSX":

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)
            dataset_2, labels, labels_list = load_images(args.dataset_2, 224)

        elif args.feature_map_1:

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)

        elif args.feature_map_2 == "CORF3D" or "TEMP3D":

            dataset_1, labels, labels_list = load_feature_maps(args.dataset_2, 224)

        elif args.feature_map_1 and args.feature_map_2 == "CORF3D" or "TEMP3D":

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)
            dataset_2, labels, labels_list = load_feature_maps(args.dataset_2, 224)

        binarizelabels = binarize_labels(labels)

    skf = train_test_split(5)
    all_fold_accuracy = []
    all_fold_loss = []
    all_fold_accuracy_1 = []
    all_fold_loss_1 = []
    all_fold_accuracy_2 = []
    all_fold_loss_2 = []
    counter = 1
    model = Models(args.trainable, args.classes, args.pretrained_dataset, args.include_top,
                   args.learning_rate)

    if args.model == 'densenet121':
        model.densenet121()
    elif args.model == 'mobilenet':
        model.mobilenet()
    elif args.model == 'xception':
        model.xception()

    model.model_trainable()
    compiled_model = model.model_compile(args.learning_rate)

    if args.method == "5_fold":

        X_train_index, X_test_index, y_train_index, y_test_index = five_cross_validation(dataset_1,
                                                                                         binarizelabels,
                                                                                         skf)

        if args.mode == "single":

            for i in range(0, len(X_train_index)):
                X_train, X_test = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train, y_test = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model, hist, loss, accuracy = train_model(compiled_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model, counter)

                plot_data_graph(hist, args.num_epochs, counter, args.model)
                pred = model_predictions(model, X_test)
                printWrongPredictions(pred, y_test, labels)
                counter = counter + 1
                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

        if args.mode == "fusion":

            for i in range(0, len(X_train_index)):
                X_train_1, X_test_1 = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train_1, y_test_1 = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                X_train_2, X_test_2 = dataset_2[X_train_index[i]], dataset_2[X_test_index[i]]
                y_train_2, y_test_2 = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs, X_test_1,
                                                                  y_test_1, args.model, counter)

                model_2, hist_2, loss_2, accuracy_2 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs, X_test_1,
                                                                  y_test_1, args.model, counter)

                plot_data_graph(hist_1, args.num_epochs, counter, args.model)
                pred_1 = model_predictions(model_1, X_test_1)
                printWrongPredictions(pred_1, y_test_1, labels)
                all_fold_accuracy_1.append(accuracy_1 * 100)
                all_fold_loss_1.append(loss_1)

                plot_data_graph(hist_2, args.num_epochs, counter, args.model)
                pred_2 = model_predictions(model_2, X_test_2)
                printWrongPredictions(pred_2, y_test_2, labels)
                all_fold_accuracy_2.append(accuracy_2 * 100)
                all_fold_loss_2.append(loss_2)

                # Convert to single column
                y_train = np.argmax(y_train_1, axis=1)
                y_test = np.argmax(y_test_1, axis=1)

                # Extract features
                model_1_feature_map = rgb_msx_feature_map(model_1, args.model, counter)
                model_2_feature_map = rgb_msx_feature_map(model_2, args.model, counter)

                features_train_1 = model_1_feature_map.predict(X_train_1)
                features_test_1 = model_1_feature_map.predict(X_test_1)

                features_train_2 = model_2_feature_map.predict(X_train_2)
                features_test_2 = model_2_feature_map.predict(X_test_2)

                combined_feature_train = feature_fusion(features_train_1, features_train_2)
                combined_feature_test = feature_fusion(features_test_1, features_test_2)

                clf = svm.SVC(kernel='linear')

                clf.fit(combined_feature_train, y_train)

                # Test the model
                y_pred = clf.predict(combined_feature_test)

                print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

                counter = counter + 1

    elif args.method == "leave_one_day_out":

        train_list, test_list = leave_one_day_out(args.timestamp)

        if args.mode == "single":

            for i in range(0, 9):
                X_train, X_test = dataset_1[train_list[i]], dataset_1[test_list[i]]
                y_train, y_test = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                model, hist, loss, accuracy = train_model(compiled_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model, counter)

                plot_data_graph(hist, args.num_epochs, counter, args.model)
                pred = model_predictions(model, X_test)
                printWrongPredictions(pred, y_test, labels)
                counter = counter + 1
                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

        if args.mode == "fusion":

            for i in range(0, 9):
                X_train_1, X_test_1 = dataset_1[train_list[i]], dataset_1[test_list[i]]
                y_train_1, y_test_1 = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                X_train_2, X_test_2 = dataset_2[train_list[i]], dataset_2[test_list[i]]
                y_train_2, y_test_2 = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs,
                                                                  X_test_1, y_test_1,
                                                                  args.model, counter)

                model_2, hist_2, loss_2, accuracy_2 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs, X_test_1,
                                                                  y_test_1, args.model,
                                                                  counter)

                # Convert to single column
                y_train = np.argmax(y_train_1, axis=1)
                y_test = np.argmax(y_test_1, axis=1)

                # Extract features
                model_1_feature_map = rgb_msx_feature_map(model_1, args.model, counter)
                model_2_feature_map = rgb_msx_feature_map(model_2, args.model, counter)

                features_train_1 = model_1_feature_map.predict(X_train_1)
                features_test_1 = model_1_feature_map.predict(X_test_1)

                features_train_2 = model_2_feature_map.predict(X_train_2)
                features_test_2 = model_2_feature_map.predict(X_test_2)

                combined_feature_train = feature_fusion(features_train_1, features_train_2)
                combined_feature_test = feature_fusion(features_test_1, features_test_2)

                clf = svm.SVC(kernel='linear')

                clf.fit(combined_feature_train, y_train)

                # Test the model
                y_pred = clf.predict(combined_feature_test)

                print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

                counter = counter + 1
