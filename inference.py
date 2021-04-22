import numpy as np
from cv2 import cv2
import os
from utils import get_bboxes, dilate_image, remove_noise
import glob
import argparse
import neuralgym as ng

import tensorflow as tf
from inpaint_model import InpaintCAModel

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

CHECKPOINT_DIR = './model_logs'
INPUT_SIZE = 512  # input image size for Generator

W = 512
# W = 680
H = 512


def inference(image, mask):
    # TODO: reshape to W and H if needed

    FLAGS = ng.Config('inpaint.yml')
    # print(FLAGS)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(tf.float32, shape=(1, H, W * 2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph, reuse=tf.AUTO_REUSE)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(CHECKPOINT_DIR, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)

    # TODO: change
    mask = np.dstack([mask] * 3)

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    # load pretrained model
    output_image = sess.run(output, feed_dict={input_image_ph: input_image})
    output_image = output_image[0][:, :, ::-1]
    return output_image


def sort(str_lst):
    return [s for s in sorted(str_lst)]


def read_paths(args):
    paths_image = glob.glob(args.dataset + '/*_hdrnet.jpg')
    paths_mask = glob.glob(args.dataset + '/*_inpainted_mask.png')
    return sort(paths_image), sort(paths_mask)


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


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset = '/Users/henrygabrielyan/Desktop/projects/g360/datasets/fetch1'
    # args.images = '/Users/henrygabrielyan/Desktop/projects/g360/generative_inpainting/image'
    # args.masks = '/Users/henrygabrielyan/Desktop/projects/g360/generative_inpainting/mask'
    args.output_dir = './output-pad-200'  # output directory
    # args.comparison_dir = './compare'  # output directory

    paths_image, paths_mask = read_paths(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if not os.path.exists(args.comparison_dir):
    #     os.makedirs(args.comparison_dir)

    for path_image, path_mask in zip(paths_image, paths_mask):
        print(path_image, path_mask)
        # raw mask bg 0, fg 1
        raw_mask = cv2.imread(path_mask)
        # TODO: change
        raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_RGB2BGR)

        # convert mask to grayscale and threshold
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)

        if cv2.findNonZero(mask) is None:
            print("image doesn't have non-zero pixels")
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("image doesn't have any contours")
            continue

        # get bounding boxes and erase small masks
        bboxes, mask = get_bboxes(contours=contours, mask=mask)

        image = cv2.imread(path_image)
        image_y = image.shape[0]
        image_x = image.shape[1]

        for bbox in bboxes:
            x, y, w, h = bbox

            # ========= CROPPING CAMERA =======================
            # calculate crop size as closest integer of [max(camera_w, camera_h) + padding] -> reshape(512,512)
            padding = 200 #int(max(w, h) * 0.5)
            crop_size = find_closest_dividend(max(w, h) + padding)
            print(f'Crop size {crop_size}')
            # continue

            # since we want bbox to be in center of crop, we need to calculate same crop padding to each sides of it
            crop_add_left = crop_add_right = (crop_size - w) // 2
            if (crop_size - w) % 2 != 0:
                crop_add_right += 1

            crop_add_top = crop_add_bottom = (crop_size - h) // 2
            if (crop_size - h) % 2 != 0:
                crop_add_bottom += 1

            # it could be bbox is to close to image edges, take residual crop from other side
            if x < crop_add_left:
                crop_add_right += crop_add_left - x
                crop_add_left = x
            elif x + w + crop_add_right > image_x:
                crop_add_left += x + w + crop_add_right - image_x
                crop_add_right = image_x

            if y < crop_add_top:
                crop_add_bottom += crop_add_top - y
                crop_add_top = y
            elif y + h + crop_add_bottom > image_y:
                crop_add_top += y + h + crop_add_bottom - image_y
                crop_add_bottom = image_y

            # since our pano is very big, we crop from it without resizing as in official source
            mask_large = mask[y - crop_add_top: y + h + crop_add_bottom, x - crop_add_left: x + w + crop_add_right]
            image_large = image[y - crop_add_top: y + h + crop_add_bottom, x - crop_add_left: x + w + crop_add_right,
                          :]

            # TODO: change
            # # normalize
            # # mask_large = mask_large.astype(np.float32) / 255.
            # mask_large = mask_large.astype(np.float32) / 255.
            # image_large = image_large.astype(np.float32)

            # downsample
            image_512 = cv2.resize(image_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = cv2.resize(mask_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = np.expand_dims(mask_512, axis=2)

            output_512 = inference(image_512, mask_512)
            filename = args.output_dir + '/' + os.path.splitext(os.path.basename(path_image))[0] + '_compare.jpg'
            cv2.imwrite(filename, output_512)


if __name__ == "__main__":
    main()
