import argparse
import json
import os
import pathlib
import sys

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage import exposure
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str,
                        help="Path to directory containing all images (no subdirectories)")
    parser.add_argument("output_directory", type=str,
                        help="Path to directory to save all extraction images")
    parser.add_argument("-r", "--resize", type=int, help="Size of output crop")
    return parser.parse_args()


def wait_for_imshow_close():
    """
    Wait on key press to close all windows
    """
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morph_close(image, kernel_size=20):
    """
    Apply morphological closing to fill areas
    """

    kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
    return image_closed


def morph_open(image, kernel_size=75):
    """
    Apply morphological opening to remove limbs
    """

    kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
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
    Rotates an image (degrees in degrees) and expands image to avoid cropping
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


def is_top_heavy(image):
    """
    Fix orientation such that the frog's head is in the top half and the frog's legs in the bottom
    To do this, we assume that the half with the most white pixels is the half with legs
    """

    img_top = image[:image.shape[0]//2]
    img_bottom = image[image.shape[0]//2:]
    return np.count_nonzero(img_top) > np.count_nonzero(img_bottom)


def get_one_image(images):
    """
    Concatenate multiple images of different resolutions vertically
    """

    heights = [x.shape[0] for x in images]
    widths = [x.shape[1] for x in images]

    merged_image = np.zeros((max(heights), sum(widths), 3), dtype=np.uint8)
    # merged_image[:, :, :] = 0

    if len(images[0].shape) == 2:
        img_color = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
    else:
        img_color = images[0]
    merged_image[:heights[0], :widths[0], :] = img_color

    for i, image in enumerate(images[1:]):
        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            print(i)
            img_color = image
        merged_image[:heights[i+1], sum(widths[:i+1]):sum(widths[:i+2])] = img_color

    return merged_image


def get_center_square(img: np.ndarray, margin=0.1):
    """
    Returns a square crop of the center of an image
    """

    if not 0 <= margin < 1:
        raise ValueError(f"Margin '{margin}' has to be in the range [0,1)")

    height, width = img.shape
    smallest_edge = min(height, width)
    crop_width = int(smallest_edge*(1-2*margin))
    crop_y0 = int((height-smallest_edge)/2 + margin*smallest_edge)
    crop_x0 = int((width-smallest_edge)/2 + margin*smallest_edge)
    return img[crop_y0:crop_y0+crop_width, crop_x0:crop_x0+crop_width]


def get_bottom_square(img: np.ndarray, margin=0.1):
    height, width = img.shape[:2]
    crop_width = int((1-2*margin)*min(height, width))
    crop_x_slice = slice(int(margin*width), int(margin*width)+crop_width)
    crop_y_slice = slice(height-crop_width-int(margin*crop_width), height-int(margin*crop_width))
    return img[crop_y_slice, crop_x_slice]


def draw_label(img, text, pos=(2, 2)):
    """
    https://stackoverflow.com/questions/54607447/opencv-how-to-overlay-text-on-video
    Add a text label on top of an image
    """

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    return cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def fit_to_screen(image, max_height=1080, max_width=1920):
    """
    Helper function for debugging to be used in combination with cv2.imshow()
    Scales down image to fit the screen if necessary
    """

    resize_factor = min(max_height/image.shape[0], max_width/image.shape[1])
    if resize_factor < 1:
        dim = (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
        return cv2.resize(image, dim)
    else:
        return image


def symmetry_score(image: np.ndarray):
    left_split = image.shape[1]//2
    right_split = left_split + int(image.shape[1] % 2 == 1)  # Skip middle column if width is odd

    left_half = image[:, :left_split]  # type: np.ndarray
    right_half = cv2.flip(image[:, right_split:], 1)  # type: np.ndarray

    return np.sum(left_half == right_half)/left_half.size


def apply_threshold_range(image, threshold_max, threshold_min):
    img_thresh = np.ones(image.shape)
    img_thresh[image < threshold_min] = 0
    img_thresh[image > threshold_max] = 0
    img_thresh[img_thresh > 0] = 255
    return img_thresh


def apply_contrast(image, contrast):
    alpha_c = 131*(contrast + 127)/(127*(131-contrast))
    gamma_c = 127*(1-alpha_c)

    return cv2.addWeighted(image, alpha_c, image, 0, gamma_c)


def get_opened_region(image: np.ndarray, threshold: int, show_images=False):
    # Apply thresholding on the green channel
    # img_contrast = apply_contrast(image, 60)
    img_thresh = apply_threshold_range(image, threshold, 20)

    # img_thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    # Close zeroed holes from thresholding
    img_closed = morph_close(img_thresh)

    if show_images:
        cv2.imshow('closed', img_closed)

    # Get image of biggest region by area
    biggest_region = get_biggest_region(img_closed)
    bbox_closed = biggest_region.bbox
    img_region = biggest_region.image.astype(np.uint8)*255

    # Rotate image using skimage region 'orientation' property
    rotation = 360 - np.degrees(biggest_region.orientation)
    img_rotated = rotate_image(img_region, rotation)

    if show_images:
        cv2.imshow('img_rotated', img_rotated)

    # Crop to bounding box
    biggest_region = get_biggest_region(img_rotated)
    bbox_rotated = biggest_region.bbox
    img_rotated_cropped = biggest_region.image.astype(np.uint8)*255

    if show_images:
        cv2.imshow('img_rotated_cropped', img_rotated_cropped)

    # Fix frog position
    needs_flipping = is_top_heavy(img_rotated_cropped)
    if needs_flipping:
        img_fixed = cv2.flip(img_rotated_cropped, -1)
    else:
        img_fixed = img_rotated_cropped

    if show_images:
        cv2.imshow('img_flipped', img_fixed)

    img_opened = morph_open(img_fixed)

    if show_images:
        cv2.imshow('img_opened', img_opened)
        # # Wait on key press to close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    biggest_region = get_biggest_region(img_opened)
    return biggest_region, bbox_closed, rotation, bbox_rotated, needs_flipping


def plot_image_histogram(image: np.ndarray):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def extract_square_symimage_path(image_path: str, min_threshold=80, max_threshold=80, min_symmetry=.9):
    print(image_path)
    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract green channel
    img_green = img[:, :, 1]

    threshold_scores = []
    for th in range(min_threshold, max_threshold, 5):  # old threshold value: 127
        try:
            biggest_region, bbox_closed, rotation, bbox_rotated, needs_flipping = get_opened_region(img_green, th,
                                                                                                    False)
        except ValueError:  # Encountered black images with no regions
            continue
        bbox_opened = biggest_region.bbox
        img_region_opened = biggest_region.image.astype(np.uint8)*255

        sym_score = symmetry_score(img_region_opened)
        if sym_score >= min_symmetry:
            break
        threshold_scores.append((sym_score, th))

    else:  # No symmetry score of >= min_symmetry was found, take highest symmetry score with highest threshold
        if len(threshold_scores) == 0:
            raise ValueError("Image was empty for all thresholds")
        th = sorted(threshold_scores, key=lambda x: (x[0], -x[1]), reverse=True)[0][1]
        biggest_region, bbox_closed, rotation, bbox_rotated, needs_flipping = get_opened_region(img_green, th,
                                                                                                show_images=False)
        bbox_opened = biggest_region.bbox
        img_region_opened = biggest_region.image.astype(np.uint8)*255

    img_gray_region = crop_image_to_bbox(img_gray, bbox_closed)
    img_gray_rotated = rotate_image(img_gray_region, rotation)
    img_gray_rotated_cropped = crop_image_to_bbox(img_gray_rotated, bbox_rotated)
    if needs_flipping:
        img_gray_fixed = cv2.flip(img_gray_rotated_cropped, -1)
    else:
        img_gray_fixed = img_gray_rotated_cropped
    img_gray_region_opened = crop_image_to_bbox(img_gray_fixed, bbox_opened)

    img_gray_mask = cv2.bitwise_and(img_gray_region_opened, img_gray_region_opened, mask=img_region_opened)

    height, width = img_gray_mask.shape[:2]
    crop_width = int(.8*min(height, width))
    crop_x_slice = slice(int(.1*width), int(.1*width)+crop_width)
    crop_y_slice = slice(height-crop_width-int(.1*crop_width), height-int(.1*crop_width))
    img_gray_mask_square = img_gray_mask[crop_y_slice, crop_x_slice]
    img_gray_100p = cv2.resize(img_gray_mask_square, (100, 100))
    img_eq_100p = exposure.equalize_hist(img_gray_100p)

    if np.max(img_eq_100p) <= 1:
        return (img_eq_100p*255).astype(np.uint8)
    else:
        return img_eq_100p.astype(np.uint8)


def extract_square(image_path: str, threshold=80, edge_length=100):
    print(image_path)
    # Read and resize the image
    img = cv2.imread(image_path)
    height_full, width_full = img.shape[:2]
    img_small = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    height_small, width_small = img_small.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract green channel
    img_green = img_small[:, :, 1]

    biggest_region, bbox_closed, rotation, bbox_rotated, needs_flipping = get_opened_region(img_green, threshold,
                                                                                            False)
    bbox_opened = biggest_region.bbox
    img_region_opened = biggest_region.image.astype(np.uint8)*255

    bbox_closed_full = np.multiply(bbox_closed, height_full/height_small).astype(int)
    bbox_rotated_full = np.multiply(bbox_rotated, bbox_closed_full[3]/bbox_closed[3]).astype(int)
    bbox_opened_full = np.multiply(bbox_opened, bbox_rotated_full[3]/bbox_rotated[3]).astype(int)

    img_gray_region = crop_image_to_bbox(img_gray, bbox_closed_full)
    img_gray_rotated = rotate_image(img_gray_region, rotation)
    img_gray_rotated_cropped = crop_image_to_bbox(img_gray_rotated, bbox_rotated_full)
    if needs_flipping:
        img_gray_fixed = cv2.flip(img_gray_rotated_cropped, -1)
    else:
        img_gray_fixed = img_gray_rotated_cropped
    img_gray_region_opened = crop_image_to_bbox(img_gray_fixed, bbox_opened_full)
    full_size = (img_gray_region_opened.shape[1], img_gray_region_opened.shape[0])
    img_region_opened_full = cv2.resize(img_region_opened, full_size, 0, 0, interpolation=cv2.INTER_NEAREST)

    img_gray_mask = cv2.bitwise_and(img_gray_region_opened, img_gray_region_opened, mask=img_region_opened_full)
    img_gray_mask_square = get_center_square(img_gray_mask, margin=.2)
    # if img_gray_mask_square.shape[0] < edge_length:
    #     print(f"Warning: Crop is being upscaled from {img_gray_mask_square.shape[0]} to {edge_length}")

    n_zero = np.count_nonzero(img_gray_mask_square == 0)
    if n_zero/img_gray_mask_square.size > 0.1:
        # cv2.imshow('square', img_gray_resized)
        # wait_for_imshow_close()
        raise ValueError(f"The extracted crop has too many zero elements ({n_zero/img_gray_mask_square.size*100:.2f}%)")
    img_eq = exposure.equalize_hist(img_gray_mask_square)

    # If image uses [0,1] floats, convert to [0,255] ints
    if np.max(img_eq) <= 1:
        return (img_eq*255).astype(np.uint8)
    else:
        return img_eq.astype(np.uint8)


def main():
    args = get_args()

    pathlib.Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    failed_extractions = []
    n_images = 0

    for image in os.listdir(args.input_directory):
        # TODO: support more image formats by converting to cv2-supported formats
        if os.path.splitext(image)[1].lower() in [".jpg", ".jpeg", ".png"]:
            n_images += 1
            full_path = os.path.join(args.input_directory, image)
            try:
                extraction_image = extract_square(full_path)
            except (ValueError, cv2.error) as e:
                print(f'Encountered an exception during extraction of \"{full_path}\":\n{e}', file=sys.stderr)
                failed_extractions.append(full_path)
            else:
                cv2.imwrite(os.path.join(args.output_directory, os.path.splitext(image)[0] + ".png"), extraction_image)

    print(f"{len(failed_extractions)} missing out of {n_images}.")
    with open(os.path.join(args.output_directory, "missing.txt"), 'w') as f:
        f.write("\n".join(failed_extractions))


def write_progress_images():
    out_directory = "pred_incorrect_steps_221121"
    in_directory = "data/Tripod"
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for file in os.listdir(in_directory):
        if not file.startswith("221121"):
            continue
        img = cv2.imread(os.path.join(in_directory, file))
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        img_green = img[:, :, 1]
        th = 80
        biggest_region, bbox_closed, rotation, bbox_rotated, needs_flipping = get_opened_region(img_green, th, False)
        bbox_opened = biggest_region.bbox
        img_region_opened = biggest_region.image.astype(np.uint8)*255
        img_thresh = apply_threshold_range(img_green, 80, 20)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_region = crop_image_to_bbox(img_gray, bbox_closed)
        img_gray_rotated = rotate_image(img_gray_region, rotation)
        img_gray_rotated_cropped = crop_image_to_bbox(img_gray_rotated, bbox_rotated)
        if needs_flipping:
            img_gray_fixed = cv2.flip(img_gray_rotated_cropped, -1)
        else:
            img_gray_fixed = img_gray_rotated_cropped
        img_gray_region_opened = crop_image_to_bbox(img_gray_fixed, bbox_opened)

        # img_eq = exposure.equalize_hist(img_gray_region_opened)

        img_gray_mask = cv2.bitwise_and(img_gray_region_opened, img_gray_region_opened, mask=img_region_opened)

        height, width = img_gray_mask.shape[:2]
        crop_width = int(.8*min(height, width))
        crop_x_slice = slice(int(.1*width), int(.1*width)+crop_width)
        crop_y_slice = slice(height-crop_width-int(.1*crop_width), height-int(.1*crop_width))
        img_gray_mask_square = img_gray_mask[crop_y_slice, crop_x_slice]
        img_gray_100p = cv2.resize(img_gray_mask_square, (100, 100))

        # cv2.imshow('gray_100p', img_gray_100p)

        img_eq_100p = exposure.equalize_hist(img_gray_100p)
        # vals = img_gray_100p.flatten()
        # b, bins, patches = plt.hist(vals, 10)
        # plt.xlim([0, 255])
        # plt.show()

        # Wait on key press to close all windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if np.max(img_eq_100p) <= 1:
            img_eq_100p = (img_eq_100p*255).astype(np.uint8)
        else:
            img_eq_100p = img_eq_100p.astype(np.uint8)
        one_image = fit_to_screen(get_one_image([img, img_thresh, img_gray_mask, img_eq_100p]))
        cv2.imwrite(os.path.join(out_directory, os.path.splitext(file)[0] + ".png"), one_image)


def angle_between_points(p1, p2):
    d1 = p2[0] - p1[0]
    d2 = p2[1] - p1[1]
    if d1 == 0:
        if d2 == 0:  # same points?
            deg = 0
        else:
            deg = 0 if p1[1] > p2[1] else 180
    elif d2 == 0:
        deg = 90 if p1[0] < p2[0] else 270
    else:
        deg = np.arctan(d2 / d1) / np.pi * 180
        lowering = p1[1] < p2[1]
        if (lowering and deg < 0) or (not lowering and deg > 0):
            deg += 270
        else:
            deg += 90
    return deg


def calc_coords_after_rotation(point, image_size, image_rot_size, degrees):
    img_height, img_width = image_size
    img_rot_height, img_rot_width = image_rot_size
    point_from_centre = np.array([point[0] - img_width/2, point[1] - img_height/2]) + .5
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    point_rot_from_centre = R.dot(point_from_centre)
    point_rot_x = point_rot_from_centre[0] + img_width/2 + (img_rot_width-img_width)/2
    point_rot_y = point_rot_from_centre[1] + img_height/2 + (img_rot_height-img_height)/2
    point_rot = np.array([point_rot_x, point_rot_y], dtype=np.int)
    return point_rot


def angle_from_line(p0, p1):
    angle = np.degrees(np.arctan2(np.abs(p0[1]-p1[1]), np.abs(p1[0]-p0[0])))
    if p1[0] > p0[0]:  # right
        if p1[1] < p0[1]:  # top-right
            angle = 90.0 - angle
        else:  # bottom-right
            angle += 90.0
    elif p1[1] > p0[1]:  # bottom-left
        angle = 270.0 - angle
    else:  # bottom-right
        angle += 270.0
    angle %= 360.0
    return angle


def extract_from_line(image_path, head_xy, tail_xy, ignore_orientation=True):
    angle = angle_from_line(tail_xy, head_xy)
    cv2_flags = cv2.IMREAD_COLOR
    if ignore_orientation:
        cv2_flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imread(image_path, cv2_flags)
    img_gray = rotate_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), angle)

    # Extract green channel
    img_green = rotate_image(img[:, :, 1], angle)

    head_rot_xy = calc_coords_after_rotation(head_xy, img.shape[:2], img_green.shape[:2], -angle)
    tail_rot_xy = calc_coords_after_rotation(tail_xy, img.shape[:2], img_green.shape[:2], -angle)

    frog_length = tail_rot_xy[1]-head_rot_xy[1]
    edge_length = int(round(frog_length*0.35))
    crop_y = int((head_rot_xy[1]+tail_rot_xy[1])/2 - .5*edge_length)
    crop_x = head_rot_xy[0] - edge_length//2
    img_crop = img_gray[crop_y:crop_y+edge_length, crop_x:crop_x+edge_length]

    img_eq = exposure.equalize_hist(img_crop)
    if np.max(img_eq) <= 1:
        return (img_eq*255).astype(np.uint8)
    else:
        return img_eq.astype(np.uint8)


def extraction_from_line():
    args = get_args()

    pathlib.Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    n_images = 0
    failed_extractions = []

    with open('head_tail_coords/coords.json', 'r') as f:
        photos = json.load(f)

    for photo in photos:
        print(photo['name'])
        full_path = os.path.join(args.input_directory, photo['name'])
        head_xy = [photo['x'][0], photo['y'][0]]
        tail_xy = [photo['x'][1], photo['y'][1]]
        n_images += 1
        try:
            extraction_image = extract_from_line(full_path, head_xy, tail_xy)
            if args.resize:
                extraction_image = cv2.resize(extraction_image, (args.resize, args.resize))
        except (ValueError, cv2.error) as e:
            print(f'Encountered an exception during extraction of \"{full_path}\":\n{e}', file=sys.stderr)
            failed_extractions.append(full_path)
        else:
            cv2.imwrite(os.path.join(args.output_directory, os.path.splitext(photo['name'])[0] + ".png"),
                        extraction_image)

    print(f"{len(failed_extractions)} missing out of {n_images}.")
    with open(os.path.join(args.output_directory, "missing.txt"), 'w') as f:
        f.write("\n".join(failed_extractions))


def extraction_test():
    # broken = ["220728_frog1_tank1_DT_PocoF3_trip.jpg", "220728_frog5_tank1_DT_MiMix2_trip.jpg",
    #           "220801_frog2_tank2_JLK_iPhoneSE_trip.jpg", "220801_frog6_tank2_JLK_iPhoneSE_trip.jpg",
    #           "220803_frog6_tank3_JLK_iPhoneSE_trip.jpg", "220803_frog18_tank2_DT_MiMix2_trip(3).jpg",
    #           "220804_frog7_tank3_DT_MiMix2_trip(3).jpg"]
    broken = ["220728_frog10_tank1_DT_MiMix2_trip.jpg"]
    working = ["220728_frog1_tank1_DT_MiMix2_trip.jpg"]

    dir_images = os.path.join(".", "data", "Tripod")

    for img_name in broken:
        img = cv2.imread(os.path.join(dir_images, img_name))
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        img = apply_contrast(img, 30)
        img_green = img[:, :, 1]
        img_thresh = apply_threshold_range(img_green, 80, 10)
        img_close = morph_close(img_thresh)
        cv2.imshow('close', img_close)

        # square = extract_square(os.path.join(dir_images, img_name))
        # cv2.imshow('square', square)
        break

    # Wait on key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extraction_test2():
    dir_images = os.path.join(".", "data", "Tripod")
    for image_file in os.listdir(dir_images):
        if os.path.splitext(image_file)[1].lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imread(os.path.join(dir_images, image_file))
        img_green = img[:, :, 1]
        img_blue = img[:, :, 0]
        img_green[img_blue > img_green] = 0
        img_thresh = apply_threshold_range(img_green, 80, 5)

        img_closed = morph_close(img_thresh, 100)
        biggest_region = get_biggest_region(img_closed)
        bbox_closed = biggest_region.bbox
        img_region = biggest_region.image.astype(np.uint8)*255

        img_opened = morph_open(img_region, 500)
        biggest_region = get_biggest_region(img_opened)
        bbox_opened = biggest_region.bbox
        img_region = biggest_region.image.astype(np.uint8)*255
        cv2.imshow(image_file, fit_to_screen(img_region))
        wait_for_imshow_close()


if __name__ == '__main__':
    extraction_from_line()
