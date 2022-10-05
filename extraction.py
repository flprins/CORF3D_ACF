import argparse
import os

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import exposure


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

    kernel_close = np.ones((8, 8), np.uint8)
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


def fix_orientation(image, image_gray):
    """
    Fix orientation such that the frog's head is in the top half and the frog's legs in the bottom
    To do this, we assume that the half with the most white pixels is the half with legs
    """
    img_top = image[:image.shape[0]//2]
    img_bottom = image[image.shape[0]//2:]
    if np.count_nonzero(img_top) > np.count_nonzero(img_bottom):
        # Rotate 180 degrees by flipping on both axes
        return cv2.flip(image, -1), cv2.flip(image_gray, -1)
    else:
        return image, image_gray


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
    # old threshold value: 127
    img_thresh = cv2.threshold(img_green, 105, 255, cv2.THRESH_BINARY_INV)[1]

    # Close zeroed holes from thresholding
    img_closed = morph_close(img_thresh)

    # Get image of biggest region by area
    biggest_region = get_biggest_region(img_closed)
    bbox_closed = biggest_region.bbox
    img_region = biggest_region.image.astype(np.uint8)*255

    img_gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray2bbox = crop_image_to_bbox(img_gray_full, bbox_closed)
    img_region_gray = cv2.bitwise_and(img_gray2bbox, img_gray2bbox, mask=img_region)

    rotation = 360 - np.degrees(biggest_region.orientation)
    img_rotated = rotate_image(img_region_gray, rotation)

    biggest_region = get_biggest_region(rotate_image(img_region, rotation))
    bbox_rotated = biggest_region.bbox
    img_rotated_cropped = biggest_region.image.astype(np.uint8)*255

    # Fix frog position
    print('rotated_cropped', img_rotated_cropped.shape)
    img_region = fix_orientation(img_rotated_cropped)
    print('img_region', img_region.shape)

    img_opened = morph_open(img_region)
    biggest_region = get_biggest_region(img_opened)
    bbox_opened = biggest_region.bbox
    img_region_opened = biggest_region.image.astype(np.uint8)*255

    img_gray2bbox = crop_image_to_bbox(crop_image_to_bbox(crop_image_to_bbox(img_gray_full, bbox_closed), bbox_rotated),
                                       bbox_opened)

    print(img_region_opened.shape)
    print(img_gray2bbox.shape)

    img_gray_region = cv2.bitwise_and(img_gray2bbox, img_gray2bbox, mask=img_region_opened)

    img_eq = exposure.equalize_hist(img_gray_region)
    # cv2.imshow('eq', img_eq)

    # rotation = 360 - np.degrees(biggest_region.orientation)
    # img_rotated = rotate_image(img_gray_region, rotation)

    # Wait on key press to close all windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    merged_image = get_one_image([img_gray_full, img_closed, img_opened, img_rotated])
    return merged_image


def extract_square(image_path: str):
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding on the green channel
    img_green = img[:, :, 1]
    # old threshold value: 127
    img_thresh = cv2.threshold(img_green, 105, 255, cv2.THRESH_BINARY_INV)[1]

    # Close zeroed holes from thresholding
    img_closed = morph_close(img_thresh)

    # Get image of biggest region by area
    biggest_region = get_biggest_region(img_closed)
    bbox_closed = biggest_region.bbox
    img_region = biggest_region.image.astype(np.uint8)*255
    img_gray_region = crop_image_to_bbox(img_gray, bbox_closed)

    rotation = 360 - np.degrees(biggest_region.orientation)
    img_rotated = rotate_image(img_region, rotation)
    img_gray_rotated = rotate_image(img_gray_region, rotation)

    biggest_region = get_biggest_region(img_rotated)
    bbox_rotated = biggest_region.bbox
    img_rotated_cropped = biggest_region.image.astype(np.uint8)*255
    img_gray_rotated_cropped = crop_image_to_bbox(img_gray_rotated, bbox_rotated)

    # Fix frog position
    img_fixed, img_gray_fixed = fix_orientation(img_rotated_cropped, img_gray_rotated_cropped)

    img_opened = morph_open(img_fixed)
    biggest_region = get_biggest_region(img_opened)
    bbox_opened = biggest_region.bbox
    img_region_opened = biggest_region.image.astype(np.uint8)*255
    img_gray_region_opened = crop_image_to_bbox(img_gray_fixed, bbox_opened)

    # img_eq = exposure.equalize_hist(img_gray_region_opened)

    img_gray_mask = cv2.bitwise_and(img_gray_region_opened, img_gray_region_opened, mask=img_region_opened)

    height, width = img_gray_mask.shape[:2]
    crop_width = int(.8*width)
    crop_x_slice = slice(int(.1*width), int(.1*width)+crop_width)
    crop_y_slice = slice(height-crop_width-int(.1*crop_width), height-int(.1*crop_width))
    img_gray_mask_square = img_gray_mask[crop_y_slice, crop_x_slice]
    img_gray_100p = cv2.resize(img_gray_mask_square, (100, 100))
    img_eq_100p = exposure.equalize_hist(img_gray_100p)
    # cv2.imshow('test', img_eq_100p)

    # Wait on key press to close all windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_eq_100p*255


def main():
    args = get_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    for image in os.listdir(args.input_directory):
        # TODO: support more image formats by converting to cv2-supported formats
        if os.path.splitext(image)[1].lower() in [".jpg", ".jpeg", ".png"]:
            full_path = os.path.join(args.input_directory, image)
            try:
                extraction_image = extract_square(full_path)
            except cv2.error as e:
                print(f'Encountered an exception during extraction of \"{full_path}\":\n{e}')
            else:
                cv2.imwrite(os.path.join(args.output_directory, os.path.splitext(image)[0] + ".png"), extraction_image)


def extraction_test():
    broken = ["220805_frog16_tank3_DT_PocoF3_trip(3).jpg", "220805_frog14_tank3_DT_OnePlusX_trip(3).jpg",
              "220805_frog14_tank3_DT_PocoF3_trip(3).jpg", "220805_frog11_tank3_DT_MiMix2_trip(3).jpg",
              "220805_frog11_tank3_DT_PocoF3_trip(3).jpg"]
    working = ["220728_frog1_tank1_DT_OnePlusX_trip(2).jpg"]

    dir_images = os.path.join(".", "data", "Tripod")

    for img_name in broken:
        img = cv2.imread(os.path.join(dir_images, img_name))
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

        # cv2.imshow('original', img)

        # img_blue = img[:, :, 0]
        img_green = img[:, :, 1]
        img_eq = exposure.equalize_hist(img_green)
        cv2.imshow(f'{img_name}_original', img)
        cv2.imshow(f'{img_name}_eq', img_eq)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # img_thresh = cv2.threshold(img_green, 95, 255, cv2.THRESH_BINARY_INV)[1]
        # cv2.imshow('thresh_broken', img_thresh)
        # img_closed = morph_close(img_thresh)
        # cv2.imshow(f'{img_name}_closed', img_closed)

    # img_working = cv2.imread(os.path.join(dir_images, working[0]))
    # img_working = cv2.resize(img_working, (0, 0), fx=0.2, fy=0.2)
    # img_thresh_working = cv2.threshold(img_working[:, :, 1], 95, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('thresh_working', img_thresh_working)
    # img_closed_working = morph_close(img_thresh_working)
    # cv2.imshow('closed_working', img_closed_working)

    # img_red = img[:, :, 2]
    # cv2.imshow('blue', img_blue)
    # cv2.imshow('green', img_green)
    # cv2.imshow('red', img_red)
    #
    # (h, w) = img.shape[:2]
    # print(img_blue[h//2][w//2])
    # print(img_green[h//2][w//2])
    # print(img_red[h//2][w//2])

    # Wait on key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
