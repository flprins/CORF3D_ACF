import argparse

from src.data_load import load_images, binarize_labels
from src.data_preprocessing import batch_data_preprocessing
from src.evaluation import model_predictions, printWrongPredictions
from src.feature_maps import corf_feature_maps, temp_feature_maps
from src.models import Models
from src.train import train_test_split, train_model
from src.visualizations import visualize_scatter_with_images, tsne, plot_data_graph

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    inpainted_images, labels, labels_list = batch_data_preprocessing()
    corf_feature_set = corf_feature_maps(inpainted_images)
    temp_feature_set = temp_feature_maps()
    binarizelabels = binarize_labels(labels)

    skf = train_test_split(5)
    all_fold_accuracy = []
    all_fold_loss = []
    model = Models()

    # if args.model == 'densenet121':
    #     model.densenet121()
    # elif args.model == 'mobilenet':
    #     model.mobilenet()
    # elif args.model == 'xception':
    #     model.xception()

    model.model_trainable()
    complied_model = model.model_compile()

    # if args.tsne:
    #     tsne_result_scaled, float_labels = tsne(list_of_image_paths, labels_list)
    #     visualize_scatter_with_images(tsne_result_scaled, list_of_image_paths)

    # for train_index, test_index in skf.split(list_of_image_paths, binarizelabels):
    #     X_train, X_test = list_of_image_paths[train_index], list_of_image_paths[test_index]
    #     y_train, y_test = binarizelabels[train_index], binarizelabels[test_index]
    #
    #     if args.mode == "train":
    #         model, hist, loss, accuracy = train_model(complied_model, X_train,
    #                                                   y_train, args.batch_size, args.num_epochs,
    #                                                   X_test, y_test, args.model)
    #
    #         plot_data_graph(hist, args.num_epochs, args.model)
    #         pred = model_predictions(model, X_test)
    #         printWrongPredictions(pred, y_test, labels)
    #
    #         all_fold_accuracy.append(accuracy * 100)
    #         all_fold_loss.append(loss)







