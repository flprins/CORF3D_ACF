import argparse
from src.data_preprocessing import batch_data_preprocessing
from src.feature_maps import corf_feature_maps

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    check = batch_data_preprocessing("D:\Amey\Masters\Projects\CORF3D_HCR\data\interim\Pre-segmentation")

