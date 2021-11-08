import argparse

from src.data_load import load_images, binarize_labels
from src.data_preprocessing import batch_data_preprocessing
from src.evaluation import model_predictions, printWrongPredictions
from src.feature_maps import corf_feature_maps, temp_feature_maps, feature_fusion, \
    rgb_msx_feature_map
from src.models import Models
from src.train import train_test_split, train_model, five_cross_validation, leave_one_day_out
from src.visualizations import visualize_scatter_with_images, tsne, plot_data_graph
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    # inpainted_images, labels, labels_list = batch_data_preprocessing()
    # corf_feature_set = corf_feature_maps(inpainted_images)
    # temp_feature_set = temp_feature_maps()


    dataset, labels, labels_list = load_images("D:\Amey\Masters\Projects\CORF3D_HCR\data\Raw\RGB (320 x 240)", 224)
    binarizelabels = binarize_labels(labels)
    skf = train_test_split(5)
    all_fold_accuracy = []
    all_fold_loss = []
    all_fold_accuracy_1 = []
    all_fold_loss_1 = []
    all_fold_accuracy_2 = []
    all_fold_loss_2 = []
    counter = 1
    model = Models()

    if args.model == 'densenet121':
        model.densenet121()
    elif args.model == 'mobilenet':
        model.mobilenet()
    elif args.model == 'xception':
        model.xception()

    model.model_trainable()
    complied_model = model.model_compile()

    # if args.tsne:
    #     tsne_result_scaled, float_labels = tsne(list_of_image_paths, labels_list)
    #     visualize_scatter_with_images(tsne_result_scaled, list_of_image_paths)

    if args.method == "5_fold":

        X_train_index, X_test_index, y_train_index, y_test_index = five_cross_validation(dataset,
                                                                             binarizelabels, skf)

        if args.mode == "single":

            for i in range(0, len(X_train_index)):
                X_train, X_test = dataset[X_train_index[i]], dataset[X_test_index[i]]
                y_train, y_test = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model, hist, loss, accuracy = train_model(complied_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model)

                plot_data_graph(hist, args.num_epochs, args.model)
                pred = model_predictions(model, X_test)
                printWrongPredictions(pred, y_test, labels)

                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

        if args.mode == "fusion":

            for i in range(0, len(X_train_index)):
                X_train_1, X_test_1 = dataset[X_train_index[i]], dataset[X_test_index[i]]
                y_train_1, y_test_1 = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                X_train_2, X_test_2 = dataset[X_train_index[i]], dataset[X_test_index[i]]
                y_train_2, y_test_2 = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(complied_model, X_train_1,
                                                          y_train_1, args.batch_size,
                                                          args.num_epochs,
                                                          X_test_1, y_test_1, args.model)

                model_2, hist_2, loss_2, accuracy_2 = train_model(complied_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs,
                                                                  X_test_1, y_test_1, args.model)

                model_1_feature_map = rgb_msx_feature_map(model_1, args.model, counter)
                model_2_feature_map = rgb_msx_feature_map(model_2, args.model, counter)
                combined_feature_map = feature_fusion(model_1_feature_map, model_2_feature_map)

                plot_data_graph(hist_1, args.num_epochs, args.model)
                pred_1 = model_predictions(model_1, X_test_1)
                printWrongPredictions(pred_1, y_test_1, labels)
                all_fold_accuracy_1.append(accuracy_1 * 100)
                all_fold_loss_1.append(loss_1)

                plot_data_graph(hist_2, args.num_epochs, args.model)
                pred_2 = model_predictions(model_2, X_test_2)
                printWrongPredictions(pred_2, y_test_2, labels)
                all_fold_accuracy_2.append(accuracy_2 * 100)
                all_fold_loss_2.append(loss_2)

    elif args.method == "leave_one_day_out":

        train_list, test_list = leave_one_day_out(dataset, binarizelabels, args.timestamp)

        if args.mode == "single":

            for i in range(0, 9):
                X_train, X_test = dataset[train_list[i]], dataset[test_list[i]]
                y_train, y_test = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                model, hist, loss, accuracy = train_model(complied_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model)

                plot_data_graph(hist, args.num_epochs, args.model)
                pred = model_predictions(model, X_test)
                printWrongPredictions(pred, y_test, labels)

                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

        if args.mode == "fusion":

            for i in range(0, 9):
                X_train_1, X_test_1 = dataset[train_list[i]], dataset[test_list[i]]
                y_train_1, y_test_1 = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                X_train_2, X_test_2 = dataset[train_list[i]], dataset[test_list[i]]
                y_train_2, y_test_2 = binarizelabels[train_list[i]], binarizelabels[test_list[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(complied_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs,
                                                                  X_test_1, y_test_1, args.model)

                model_2, hist_2, loss_2, accuracy_2 = train_model(complied_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs,
                                                                  X_test_1, y_test_1, args.model)

                model_1_feature_map = rgb_msx_feature_map(model_1, args.model, counter)
                model_2_feature_map = rgb_msx_feature_map(model_2, args.model, counter)
                combined_feature_map = feature_fusion(model_1_feature_map, model_2_feature_map)

                plot_data_graph(hist_1, args.num_epochs, args.model)
                pred_1 = model_predictions(model_1, X_test_1)
                printWrongPredictions(pred_1, y_test_1, labels)
                all_fold_accuracy_1.append(accuracy_1 * 100)
                all_fold_loss_1.append(loss_1)

                plot_data_graph(hist_2, args.num_epochs, args.model)
                pred_2 = model_predictions(model_2, X_test_2)
                printWrongPredictions(pred_2, y_test_2, labels)
                all_fold_accuracy_2.append(accuracy_2 * 100)
                all_fold_loss_2.append(loss_2)

