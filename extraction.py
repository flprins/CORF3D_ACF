import argparse
import os

import cv2
import numpy as np
from skimage.measure import label, regionprops


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str,
                        help="Path to directory containing all images (no subdirectories)")
    parser.add_argument("output_directory", type=str,
                        help="Path to directory to save all extraction images")
    return parser.parse_args()


def morph_close(image):
    """
    Apply morphological closing to fill areas
    """

    kernel_close = np.ones((10, 10), np.uint8)
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
    return image_closed


def morph_open(image):
    """
    Apply morphological opening to remove limbs
    """

    kernel_open = np.ones((75, 75), np.uint8)
    image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
    return image_opened


def get_biggest_region(image):
    """
    Finds all regions in a binary image and returns the biggest
    """

    # Get regions and their properties
    label_img = label(image)
    regions = regionprops(label_img)

    # Return the biggest region
    biggest_region = max(regions, key=lambda x: x.area)
    return biggest_region


def crop_image_to_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    img_cropped = image[x1:x2, y1:y2]
    return img_cropped


def rotate_image(mat: np.ndarray, angle: float):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    https://stackoverflow.com/a/37347070
    """

    # image shape has 3 dimensions
    height, width = mat.shape[:2]
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_one_image(images):
    """
    Concatenate multiple images of different resolutions vertically
    """
    heights = [x.shape[0] for x in images]
    widths = [x.shape[1] for x in images]

    merged_image = np.zeros((max(heights), sum(widths)), dtype=np.uint8)
    merged_image[:, :] = 0

    merged_image[:heights[0], :widths[0]] = images[0]

    for i, image in enumerate(images[1:]):
        merged_image[:heights[i+1], sum(widths[:i+1]):sum(widths[:i+2])] = image

    return merged_image


def plot_preprocessing(image_path: str):
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    # Apply thresholding on the green channel
    img_green = img[:, :, 1]
    img_thresh = cv2.threshold(img_green, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # Close zeroed holes from thresholding
    img_closed = morph_close(img_thresh)

    # Get image of biggest region by area
    biggest_region = get_biggest_region(img_closed)
    bbox_closed = biggest_region.bbox
    img_region = biggest_region.image.astype(np.uint8)*255

    img_opened = morph_open(img_region)

    biggest_region = get_biggest_region(img_opened)
    bbox_opened = biggest_region.bbox
    img_region_opened = biggest_region.image.astype(np.uint8)*255

    img_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray2bbox = crop_image_to_bbox(crop_image_to_bbox(img_gray_full, bbox_closed), bbox_opened)
    # cv2.imshow('gray2bbox', img_gray2bbox)

    img_gray_region = cv2.bitwise_and(img_gray2bbox, img_gray2bbox, mask=img_region_opened)
    # cv2.imshow('cropped', img_gray_region)

    rotation = 360 - np.degrees(biggest_region.orientation)
    img_rotated = rotate_image(img_gray_region, rotation)
    merged_image = get_one_image([img_gray_full, img_closed, img_opened, img_rotated])
    return merged_image

    # Wait on key press to close all windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    args = get_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    for image in os.listdir(args.input_directory):
        if os.path.splitext(image)[1].lower() in [".jpg", ".jpeg", ".png"]:
            extraction_image = plot_preprocessing(os.path.join(args.input_directory, image))
            cv2.imwrite(os.path.join(args.output_directory, os.path.splitext(image)[0] + ".png"), extraction_image)


if __name__ == '__main__':
    main()
