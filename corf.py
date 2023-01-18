import argparse
import os
import pathlib
import traceback

import cv2
import numpy as np

from src.feature_maps import corf_feature_maps, feature_stack, feature_normalization


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str,
                        help="Path to directory containing all images (no subdirectories)")
    parser.add_argument("output_directory", type=str,
                        help="Path to directory to save all extraction images")
    return parser.parse_args()


def main():
    args = get_args()

    pathlib.Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    # TODO: Investigate misclassifications
    for inhibition_factor in [0.0, 1.8, 3.6]:
        print(f'Calculating CORF feature maps for inhibition factor {inhibition_factor}...')
        corf_feature_set = corf_feature_maps(args.input_directory, 2.2, 4, inhibition_factor, 0.005)
        corf_feature_set_norm = feature_normalization(corf_feature_set)
        np.savez_compressed(os.path.join(args.output_directory, f'corf_{inhibition_factor}'),
                            data=corf_feature_set_norm)

    labels = [x for x in os.listdir(args.input_directory) if os.path.splitext(x)[1] in [".jpg", ".jpeg", ".png"]]
    np.savez_compressed(os.path.join(args.output_directory, "labels"), np.asarray(labels))


if __name__ == '__main__':
    main()
