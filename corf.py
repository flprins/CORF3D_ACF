import argparse
import os

import cv2
import numpy as np

from src.feature_maps import corf_feature_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str,
                        help="Path to directory containing all images (no subdirectories)")
    parser.add_argument("output_directory", type=str,
                        help="Path to directory to save all extraction images")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    corf_arrays = {}
    for image in os.listdir(args.input_directory):
        full_path = os.path.join(args.input_directory, image)
        try:
            corf_image = corf_feature_map(full_path, 2.2, 4, 0.0, 0.005)
        except cv2.error as e:
            print(f'Encountered an exception during corf conversion of \"{full_path}\":\n{e}')
        else:
            corf_arrays[image] = corf_image
    np.savez_compressed(os.path.join(args.output_directory, "corf"), **corf_arrays)


if __name__ == '__main__':
    main()
