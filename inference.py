import numpy as np
from cv2 import cv2
import os
import utils
import glob
import argparse
import neuralgym as ng

import tensorflow as tf
from inpaint_model import InpaintCAModel

# import receptive_field as rf


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

CHECKPOINT_DIR = './placesv2-512'
INPUT_SIZE = 512  # input image size for Generator

W = 512
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

    args.dataset = '/home/rudolfs/Desktop/camera-removal/pano'
    args.output_dir = './output'  # output directory

    paths_image, paths_mask = utils.read_paths(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for path_image, path_mask in zip(paths_image, paths_mask):
        print(path_image, path_mask)
        # raw mask bg 0, fg 1
        raw_mask = cv2.imread(path_mask)

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

        # get bounding boxes
        bboxes, mask = utils.get_bboxes(contours=contours, mask=mask)
        image = cv2.imread(path_image)
        for idx, bbox in enumerate(bboxes):
            # use image center (approximate removing of floor and ceiling)
            image_center = image[int(0.4 * image.shape[0]):int(0.82 * image.shape[0]), :]
            x, y, w, h = utils.calc_bbox_with_pad(bbox=bbox, image=image_center)

            # since our pano is very big, we crop from it without resizing as in official source
            mask_large = mask[y:y+h, x:x+w]
            image_large = image[y:y+h, x:x+w]

            # downsample
            image_512 = cv2.resize(image_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = cv2.resize(mask_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = np.expand_dims(mask_512, axis=2)

            output_512 = inference(image_512, mask_512)
            filename = args.output_dir + '/' + os.path.splitext(os.path.basename(path_image))[0] + '_compare.jpg'
            cv2.imwrite(filename, output_512)


if __name__ == "__main__":
    main()
