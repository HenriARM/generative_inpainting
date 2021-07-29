import numpy as np
import cv2
import os
import glob


def sort(str_lst):
    return [s for s in sorted(str_lst)]


def read_paths(dataset_path, image_suffix, mask_suffix):
    paths_image = glob.glob(dataset_path + '/*' + image_suffix)
    paths_mask = glob.glob(dataset_path + '/*' + mask_suffix)
    return sort(paths_image), sort(paths_mask)

def filter_small_contours(contours, min_bbox_area):
    if not contours:
        return []
    cnts = []
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        _, _, w, h = bbox
        if w * h >= min_bbox_area:
            cnts.append(cnt)
    # TODO: cnts sort by size?
    return cnts

# Return sorted contours (first with smallest area)
def get_contours(image):
    cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(cnts, key=lambda x: cv2.contourArea(x))

# Get bounding boxes from contours
def get_bboxes(contours):
    if not contours:
        return []
    bboxes = []
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        bboxes.append(bbox)
    bboxes.sort(key=lambda b: b[0])
    return bboxes

# Get bounding boxes from contours and merge them if they are within a overlap_distance in px
def merge_bboxes(): # (contours, mask, min_bbox_area, overlap_distance)
    pass
    # TODO:
    #     x, y, w, h = bbox
    #     if w * h <= min_bbox_area:
    #         # erase mask pixel
    #         mask[y:y + h, x:x + w] = 0
    #     else:
    #         bboxes.append(bbox)
    # bboxes.sort(key=lambda b: b[0])

    # # merge intersecting bounding boxes
    # merged_bboxes = []
    # while len(bboxes) > 1:
    #     overlap = bbox_overlap(b1=bboxes[0], b2=bboxes[1], overlap_distance=overlap_distance)
    #     merged_bboxes.append(overlap)

    #     _, _, overlap_w, overlap_h = overlap
    #     _, _, b1_w, b1_h = bboxes[0]
    #     # delete b1 bbox and b2 if they were merged
    #     if overlap_w != b1_w and overlap_h != b1_h:
    #         del bboxes[1]
    #     del bboxes[0]

    # # append last bbox, which could be left if it didn't intersected with other bboxes
    # if bboxes:
    #     merged_bboxes.append(bboxes.pop())
    # return merged_bboxes, mask


def bbox_overlap(b1, b2, overlap_distance):
    """
    IoU, Calculate intersection of two bounding boxes and returns overlap.
    If bboxes doesn't intersect, b1 will be returned.
    If distance between two boxes are < overlap_distance, still count as intersect
    Each bbox has x,y,w,h
    """
    x, y, w, h = b1
    b1_x1 = x
    b1_x2 = x + w
    b1_y1 = y
    b1_y2 = y + h

    x, y, w, h = b2
    b2_x1 = x
    b2_x2 = x + w
    b2_y1 = y
    b2_y2 = y + h

    # determine the coordinates of the intersection rectangle
    x_left = max(b1_x1, b2_x1)
    x_right = min(b1_x2, b2_x2)
    y_top = max(b1_y1, b2_y1)
    y_bottom = min(b1_y2, b2_y2)

    # check that intersection W and H is not 0 with overlap_distance
    if x_right + overlap_distance > x_left and y_bottom + overlap_distance > y_top:
        # count as intersection and return overlap
        x_left = min(b1_x1, b2_x1)
        x_right = max(b1_x2, b2_x2)
        y_top = min(b1_y1, b1_y1)
        y_bottom = max(b1_y2, b2_y2)
        return x_left, y_top, x_right - x_left, y_bottom - y_top
    else:
        # return same bbox
        return b1


def find_closest_dividend(dividend, input_size):
    divisor = input_size
    """
    dividend / divisor = quotient
    closest integer >= divident / 512 (INPUT_IMAGE) = any integer >= 1 (mod 512 = 0)
    :return:
    """
    if dividend % divisor == 0:
        return divident
    else:
        return ((dividend // divisor) + 1) * divisor


def calc_bbox_with_pad(bbox, image, input_size):
    # TODO: catch cases when padding is bigger than left space to crop
    # TODO: catch cases when crop_size is too big to capture, not enough space

    image_y = image.shape[0]
    image_x = image.shape[1]
    x, y, w, h = bbox
    pad = 0
    crop_size = max(input_size, max(w,h) + pad) # find_closest_dividend(max(w, h) + pad)
    # since we want bbox to be in center of crop, we need to calculate same crop padding to each sides of it
    pad_left = pad_right = (crop_size - w) // 2
    if (crop_size - w) % 2 != 0:
        pad_right += 1

    pad_top = pad_bottom = (crop_size - h) // 2
    if (crop_size - h) % 2 != 0:
        pad_bottom += 1

    # it could be bbox is to close to image edges, take residual crop from other side
    if x < pad_left:
        pad_right += pad_left - x
        pad_left = x
    elif x + w + pad_right > image_x:
        pad_left += x + w + pad_right - image_x
        pad_right = image_x

    if y < pad_top:
        pad_bottom += pad_top - y
        pad_top = y
    elif y + h + pad_bottom > image_y:
        pad_top += y + h + pad_bottom - image_y
        pad_bottom = image_y

    return x - pad_left, y - pad_top, crop_size

# def remove_noise(img):
#     # Removes small specks from the image, adds some thickness to the mask
#     # Opening - remove noise
#     kernel = np.ones((3, 3), np.uint8)
#     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
#     # # Add some thickness to mask
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
#     # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=1)
#     return img


# def dilate_image(img):
#     """
#     Expands the mask, used for finding regions of interest
#     https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
#     """
#     # kernel = np.ones((11, 11), np.uint8)
#     # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#     # Add some thickness to mask
#     kernel = np.ones((19, 19), np.uint8)
#     img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=8)
#     return img
