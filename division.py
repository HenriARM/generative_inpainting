import numpy as np
from cv2 import cv2
import os
import utils
import glob
import argparse
import neuralgym as ng
import time

import tensorflow as tf
from inpaint_model import InpaintCAModel

CHECKPOINT_DIR = './placesv2-512'
INPUT_SIZE = 256  # input image size for Generator 512
IMAGE_SUFFIX = '_hdrnet.jpg'
MASK_SUFFIX = '_inpainted_mask.png'
INPAINT_SUFFIX = '_inpainted.png'
LOCAL_CACHE= False

MIN_BBOX_AREA = 50 * 50
OVERLAP_DISTANCE = 200

def model_compile():
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
    print('Model loaded')
    return sess, input_image_ph, output    


def static_inference(sess, input_layer, output_layer, image, mask):
    mask = np.dstack([mask] * 3)

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    # load pretrained model
    output_image = sess.run(output_layer, feed_dict={input_layer: input_image})
    output_image = output_image[0][:, :, ::-1]
    return output_image


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.dataset = '/mnt/machine_learning/datasets/hm_dataset/reports/report-13-05-2021/data'
    args.dataset = '/home/rudolfs/Desktop/reports/report-13-05-2021/data'
    args.output_dir = './output'  # output directory

    paths_image, paths_mask = utils.read_paths(dataset_path=args.dataset, image_suffix=IMAGE_SUFFIX, mask_suffix=MASK_SUFFIX)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if LOCAL_CACHE is True:
        args.patch_dir = './patch'
        if not os.path.exists(args.patch_dir):
            os.makedirs(args.patch_dir)

    # compile model
    sess, input_layer, output_layer = model_compile()    

    for path_image, path_mask in zip(paths_image, paths_mask):
        print('Artifact ', path_image, path_mask)

        # raw mask bg 0, fg 1
        raw_mask = cv2.imread(path_mask)

        # convert mask to grayscale and threshold
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)
        # raw_mask = None

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

        # check is there any bbox which is too close to artifact vertical edges  
        is_close = False
        max_distance = 0
        thres = INPUT_SIZE # distance from bbox to edge, which we consider too close
        for bbox in bboxes:
            x, y, w, h = bbox
            if x < thres or x + w > image.shape[1] - thres:
                is_close = True
                # to calculate max distance from bbox to edge, understand to which edge it is near (we need to copy whole bbox)
                if x < image.shape[1] - (x + w): # left side closer
                    max_distance = max(max_distance, x + w)  
                else: # right side closer
                    max_distance = max(max_distance, image.shape[1] - x)
        # x, y, w, h = None, None, None, None
                
        # assure distance is enough for patching
        max_distance = max(max_distance, INPUT_SIZE)
        
        # add more to distance, so new edge wont touch opposite bbox
        max_distance += 2 * thres
                
        mask = np.expand_dims(mask, axis=2)

        # copy opposite edge
        if is_close is True:
            upd_image = np.zeros(shape=(image.shape[0], image.shape[1] + 2 * max_distance, image.shape[2]))
            upd_image[:, max_distance:-max_distance, :] = np.copy(image)
            upd_image[:, :max_distance ,:] = np.copy(image[:, -max_distance:, :])
            upd_image[:, -max_distance:, :] = np.copy(image[:, :max_distance, :])
            image = upd_image

            upd_mask = np.zeros(shape=(mask.shape[0], mask.shape[1] + 2 * max_distance, mask.shape[2]))
            upd_mask[:, max_distance:-max_distance, :] = np.copy(mask)
            upd_mask[:, :max_distance ,:] = np.copy(mask[:, -max_distance:, :])
            upd_mask[:, -max_distance:, :] = np.copy(mask[:, :max_distance, :])
            mask = upd_mask
    
            # adjust x coordinates of bboxes
            for i in range(len(bboxes)):
                bboxes[i] = (bboxes[i][0] + max_distance, *bboxes[i][1:])
        # upd_image = None
        # upd_mask = None

        artifact_name = os.path.splitext(os.path.basename(path_image))[0]
        for idx, bbox in enumerate(bboxes):                      
            print(f'bbox size {bbox}')
            t = time.time()                                                                                                                                                                                                                           

            x, y, w, h = bbox
            finish_x = x + w
            finish_y = y + h

            # should be smaller than model input size
            stride = INPUT_SIZE // 4

            # bbox x0, y0 will be the center of patch (if there is enough )
            px = x - INPUT_SIZE // 2
            py = y - INPUT_SIZE // 2

            i = 0
            while py < finish_y:
                while px < finish_x:
                    mask_patch = mask[py:py+INPUT_SIZE, px:px+INPUT_SIZE]
                    image_patch = image[py:py+INPUT_SIZE, px:px+INPUT_SIZE]

                    if cv2.findNonZero(mask_patch) is None:
                        px += stride
                        continue

                    if LOCAL_CACHE is True:
                        # save mask and image
                        mask_filename = os.path.join(args.patch_dir, artifact_name + f'-bbox{idx}-patch{i}-mask.jpg')
                        image_filename = os.path.join(args.patch_dir, artifact_name + f'-bbox{idx}-patch{i}-image.jpg')
                        cv2.imwrite(mask_filename, np.dstack([mask_patch.astype(np.uint8)] * 3))
                        cv2.imwrite(image_filename, image_patch)

                    # check shapes are correct
                    if image_patch[:,:,0].shape != (INPUT_SIZE, INPUT_SIZE) or \
                        mask_patch[:,:,0].shape != (INPUT_SIZE, INPUT_SIZE):
                        print('Incorrect patch size')
                        exit(-1)

                    # inference
                    inpaint_patch = static_inference(sess, input_layer, output_layer, image_patch, mask_patch)

                    if LOCAL_CACHE is True:
                        # save inpaint
                        inpaint_filename = os.path.join(args.patch_dir, artifact_name + f'-bbox{idx}-patch{i}-inpaint.jpg')
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
                    mask[py:py+INPUT_SIZE, px:px+INPUT_SIZE] = ((mask_patch - partial_mask_patch) * 255.).astype(np.uint8)

                    i += 1 
                    px += stride
                px = x - INPUT_SIZE // 2
                py += stride

        print(f'Time on one bbox {bbox} inference: {time.time() - t}') 

        # trim image back to save
        if is_close:
            image = image[:, max_distance:-max_distance, :]
            mask = mask[:, max_distance:-max_distance, :]
        filename = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path_image))[0] + INPAINT_SUFFIX)
        cv2.imwrite(filename, image)

if __name__ == "__main__":
    main()