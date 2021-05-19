import numpy as np
from cv2 import cv2
import os
import utils
import glob
import argparse
import neuralgym as ng

import tensorflow as tf
from inpaint_model import InpaintCAModel

CHECKPOINT_DIR = './placesv2-512'
INPUT_SIZE = 768  # input image size for Generator 512
IMAGE_SUFFIX = '_hdrnet.jpg'
MASK_SUFFIX = '_inpainted_mask.png'
INPAINT_SUFFIX = '_inpainted.png'


def inference(image, mask):
    FLAGS = ng.Config('inpaint.yml')
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE * 2, 3))
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


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset = '/home/rudolfs/Desktop/reports/report-13-05-2021/data-report-13-05-2021'
    args.output_dir = './deep-768in-100pad-report-13-05-2021'  # output directory

    paths_image, paths_mask = utils.read_paths(dataset_path=dataset_path, image_suffix=IMAGE_SUFFIX, mask_suffix=MASK_SUFFIX)

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

        # get bounding boxes (returned filtered mask)
        bboxes, mask = utils.get_bboxes(contours=contours, mask=mask)

        if not bboxes:
            print("image doesn't have any bboxes")
            continue

        image = cv2.imread(path_image)
        for idx, bbox in enumerate(bboxes):
            # x, y, crop_size = utils.calc_bbox_with_pad(bbox=bbox, image=image, input_size=INPUT_SIZE)
            # use image center (approximate removing of floor and ceiling)
            center_top = int(0.4 * image.shape[0])
            center_bottom = int(0.82 * image.shape[0])
            image_center = image[center_top:center_bottom, :]
            bbox_center = (bbox[0], bbox[1] - center_top, bbox[2], bbox[3])
            x, y_center, crop_size = utils.calc_bbox_with_pad(bbox=bbox_center, image=image_center, input_size=INPUT_SIZE)
            y = y_center + center_top

            # since our pano is very big, we crop from it without resizing as in official source
            mask_large = mask[y:y+crop_size, x:x+crop_size]
            image_large = image[y:y+crop_size, x:x+crop_size]

            # downsample
            image_512 = cv2.resize(image_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = cv2.resize(mask_large, (INPUT_SIZE, INPUT_SIZE))
            mask_512 = np.expand_dims(mask_512, axis=2)

            inpaint_512 = inference(image_512, mask_512)
            inpaint_512_large = cv2.resize(inpaint_512, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

            # paste the hole region to the original raw image
            mask_large = np.expand_dims(mask_large, axis=2)
            mask_large = mask_large / 255.
            output_large = inpaint_512_large * mask_large + image_large * (1. - mask_large)
            output_large = output_large.astype(np.uint8)

            # put output into pano
            image[y:y+crop_size, x:x+crop_size] = output_large


            # filename = os.path.join(args.output_dir,os.path.splitext(os.path.basename(path_image))[0] + '_inpaint.jpg')
            # cv2.imwrite(filename, output_large)

        filename = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path_image))[0] + INPAINT_SUFFIX)
        cv2.imwrite(filename, image)


if __name__ == "__main__":
    main()
