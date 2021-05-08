import numpy as np
import cv2
import os
import glob


# tmp
INPUT_SIZE = 512


def sort(str_lst):
    return [s for s in sorted(str_lst)]


def read_paths(args):
    paths_image = glob.glob(args.dataset + '/*_hdrnet.jpg')
    paths_mask = glob.glob(args.dataset + '/*_inpainted_mask.png')
    return sort(paths_image), sort(paths_mask)


# Get bounding boxes from contours and merge them if they are within a threshold in px
def get_bboxes(contours, mask):
    if not contours:
        return []

    # tmp
    MIN_BOX_AREA = 50 * 50
    THRESHOLD = 350

    bboxes = []
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        if w * h <= MIN_BOX_AREA:
            # erase mask pixel
            mask[y:y + h, x:x + w] = 0
        else:
            bboxes.append(bbox)
    bboxes.sort(key=lambda b: b[0])

    # merge intersecting bounding boxes
    merged_bboxes = []
    while len(bboxes) > 1:
        overlap = bbox_overlap(b1=bboxes[0], b2=bboxes[1], threshold=THRESHOLD)
        merged_bboxes.append(overlap)

        _, _, overlap_w, overlap_h = overlap
        _, _, b1_w, b1_h = bboxes[0]
        # delete b1 bbox and b2 if they were merged
        if overlap_w != b1_w and overlap_h != b1_h:
            del bboxes[1]
        del bboxes[0]

    # append last bbox, which could be left if it didn't intersected with other bboxes
    if bboxes:
        merged_bboxes.append(bboxes.pop())
    return merged_bboxes, mask


def bbox_overlap(b1, b2, threshold):
    """
    IoU, Calculate intersection of two bounding boxes and returns overlap.
    If bboxes doesn't intersect, b1 will be returned.
    If distance between two boxes are < threshold, still count as intersect
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

    # check that intersection W and H is not 0 with threshold
    if x_right + threshold > x_left and y_bottom + threshold > y_top:
        # count as intersection and return overlap
        x_left = min(b1_x1, b2_x1)
        x_right = max(b1_x2, b2_x2)
        y_top = min(b1_y1, b1_y1)
        y_bottom = max(b1_y2, b2_y2)
        return x_left, y_top, x_right - x_left, y_bottom - y_top
    else:
        # return same bbox
        return b1


def find_closest_dividend(dividend):
    divisor = INPUT_SIZE
    """
    dividend / divisor = quotient
    closest integer >= divident / 512 (INPUT_IMAGE) = any integer >= 1 (mod 512 = 0)
    :return:
    """
    if dividend % divisor == 0:
        return divident
    else:
        return ((dividend // divisor) + 1) * divisor


def calc_bbox_with_pad(bbox, image):
    # TODO: catch cases when padding is bigger than left space to crop
    image_y = image.shape[0]
    image_x = image.shape[1]
    x, y, w, h = bbox
    pad = 300
    crop_size = find_closest_dividend(max(w, h) + pad)  # int(max(w, h) * 0.5)
    
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

    return x - pad_left, y - pad_top, w + pad_right, h + pad_bottom

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
