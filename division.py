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
INPUT_SIZE = 256  # input image size for Generator 512
IMAGE_SUFFIX = '_hdrnet.jpg'
MASK_SUFFIX = '_inpainted_mask.png'
INPAINT_SUFFIX = '_inpainted.png'

MIN_BBOX_AREA = 50 * 50
OVERLAP_DISTANCE = 200


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

    args.dataset = '/home/rudolfs/Desktop/reports/report-13-05-2021/data'
    args.output_dir = './output'  # output directory

    paths_image, paths_mask = utils.read_paths(dataset_path=args.dataset, image_suffix=IMAGE_SUFFIX, mask_suffix=MASK_SUFFIX)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    args.patch_dir = './patch'
    if not os.path.exists(args.patch_dir):
        os.makedirs(args.patch_dir)

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
        bboxes, mask = utils.get_bboxes(contours=contours, mask=mask, min_bbox_area=MIN_BBOX_AREA, overlap_distance=OVERLAP_DISTANCE)

        if not bboxes:
            print("image doesn't have any bboxes")
            continue

        image = cv2.imread(path_image)
        artifact_name = os.path.splitext(os.path.basename(path_image))[0]
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            finish_x = x + w
            finish_y = y + h

            stride = INPUT_SIZE // 4
            # bbox x0, y0 will be the center of patch 
            px = x - INPUT_SIZE // 2
            py = y - INPUT_SIZE // 2


            i = 0
            while py < finish_y:
                while px < finish_x:
                    mask_filename = os.path.join(args.patch_dir, artifact_name + f'_{i}_mask.jpg')
                    image_filename = os.path.join(args.patch_dir, artifact_name + f'_{i}_image.jpg')
                    inpaint_filename = os.path.join(args.patch_dir, artifact_name + f'_{i}_inpaint.jpg')

                    mask_patch = mask[py:py+INPUT_SIZE, px:px+INPUT_SIZE]
                    image_patch = image[py:py+INPUT_SIZE, px:px+INPUT_SIZE]

                    if cv2.findNonZero(mask_patch) is None:
                        px += stride
                        continue

                    # inference
                    mask_patch = np.expand_dims(mask_patch, axis=2)
                    inpaint_patch = inference(image_patch, mask_patch)

                    # save mask, image and inpaint patches
                    cv2.imwrite(mask_filename, np.dstack([mask_patch.astype(np.uint8)] * 3))
                    cv2.imwrite(image_filename, image_patch)
                    cv2.imwrite(inpaint_filename, inpaint_patch)

                    # leave half of the hole to next patch to inpaint
                    mask_patch = mask_patch / 255.
                    partial_mask_patch = np.array(mask_patch, copy=True)
                    partial_mask_patch[:, -stride // 2:, :] = 0

                    # blend inpaint into output
                    output_patch = inpaint_patch * partial_mask_patch + image_patch * (1. - partial_mask_patch)
                    output_patch = output_patch.astype(np.uint8)

                    # blend output into image and remove from big mask used mask patch (invert)
                    image[py:py+INPUT_SIZE, px:px+INPUT_SIZE] = output_patch
                    
                    caution = 10
                    partial_mask_patch[:, -stride // 2 - caution:, :] = 0
                    partial_mask_patch[-caution:, :, :] = 0
                    mask[py:py+INPUT_SIZE, px:px+INPUT_SIZE] = ((mask_patch - partial_mask_patch) * 255.).astype(np.uint8)[:,:,0]

                    # cv2.imshow("mask patch", np.dstack([(mask_patch * 255.).astype(np.uint8)] * 3) )
                    # cv2.imshow("partial", np.dstack([(partial_mask_patch * 255.).astype(np.uint8)] * 3) )
                    # cv2.imshow("wtf", np.dstack([((mask_patch - partial_mask_patch) * 255.).astype(np.uint8)] * 3) )
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit(-1)

                    i += 1 
                    px += stride
                px = x - INPUT_SIZE // 2
                py += stride

        filename = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path_image))[0] + INPAINT_SUFFIX)
        cv2.imwrite(filename, image)
        exit(-1)


if __name__ == "__main__":
    main()
