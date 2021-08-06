import numpy as np
from cv2 import cv2
import os
import utils
import glob
import argparse
import neuralgymtf2 as ng
import time

import tensorflow as tf
from inpaint_model import InpaintCAModel

tf.compat.v1.disable_eager_execution()

CHECKPOINT_DIR = '/home/henri/projects/deepfill/checkpoints/mytrain'
INPUT_SIZE = 256  # input image size for Generator 512
IMAGE_SUFFIX = '_hdrnet.jpg'
MASK_SUFFIX = '_inpainted_mask.png'
INPAINT_SUFFIX = '_inpainted.jpg'
LOCAL_CACHE= False

MIN_BBOX_AREA = 50 * 50
OVERLAP_DISTANCE = 200

PATCHING_TYPE = 'CONVEX' # CONTOUR | CONV | CONVEX

def cache(patch_dir, artifact_name, bidx, pidx, image_patch, mask_patch, inpaint_patch, output_patch):
    # TODO: cidx is not unique
    # TODO: mask is not printed
    mask_filename = os.path.join(patch_dir, artifact_name + f'-bbox{bidx}-patch{pidx}-mask.jpg')
    image_filename = os.path.join(patch_dir, artifact_name + f'-bbox{bidx}-patch{pidx}-image.jpg')
    inpaint_filename = os.path.join(patch_dir, artifact_name + f'-bbox{bidx}-patch{pidx}-inpaint.jpg')
    output_filename = os.path.join(patch_dir, artifact_name + f'-bbox{bidx}-patch{pidx}-output.jpg')

    cv2.imwrite(mask_filename, mask_patch)
    cv2.imwrite(image_filename, image_patch)
    cv2.imwrite(inpaint_filename, inpaint_patch)
    cv2.imwrite(output_filename, output_patch)

def model_compile():
    FLAGS = ng.Config('inpaint.yml')
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.compat.v1.placeholder(tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE * 2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph, reuse=tf.compat.v1.AUTO_REUSE)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.train.load_variable(CHECKPOINT_DIR, from_name)
        assign_ops.append(tf.compat.v1.assign(var, var_value))
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

def mask_close_to_edges(image, bboxes):
    # check is there any bbox which is too close to vertical edges  
    is_close = False
    max_distance = 0
    thres = INPUT_SIZE # distance from bbox to edge, which we consider too close
    image_w = image.shape[1]
    for bbox in bboxes:
        x, _, w, _ = bbox
        if x < thres or x + w > image_w - thres:
            is_close = True
            # which edge it is near (we need to copy whole bbox)
            if x < image_w - (x + w): # left side closer
                max_distance = max(max_distance, x + w)  
            else: # right side closer
                max_distance = max(max_distance, image_w - x)
        # x, y, w, h = None, None, None, None
    # assure distance is enough for patching
    max_distance = max(max_distance, INPUT_SIZE)
        
    # add more to distance, so new edge wont touch opposite bbox
    max_distance += 2 * thres
    return is_close, max_distance            


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.dataset = '/mnt/machine_learning/datasets/hm_dataset/reports/report-13-05-2021/data'
    args.dataset = '/home/henri/datasets/artifacts/panos/pan-21-07-2021/data'
    args.output_dir = '/home/henri/datasets/artifacts/panos/pan-21-07-2021/mytrain-convex'  # output directory

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
        artifact_name = os.path.splitext(os.path.basename(path_image))[0]
        print(f'Artifact: {artifact_name}')

        # raw mask bg 0, fg 1
        raw_mask = cv2.imread(path_mask)
        image = cv2.imread(path_image)

        # convert mask to grayscale and threshold
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)

        if cv2.findNonZero(mask) is None:
            print("image doesn't have non-zero pixels")
            continue

        # base contours are the ones used on first patching 
        base_contours = utils.get_contours(image=mask)

        # filter out small contours
        # base_contours = utils.filter_small_contours(contours=base_contours, min_bbox_area=MIN_BBOX_AREA)

        # get bounding boxes 
        bboxes = utils.get_bboxes(contours=base_contours)
        # mask=mask, min_bbox_area=MIN_BBOX_AREA, overlap_distance=OVERLAP_DISTANCE

        # copy opposite edges if needed
        is_close, max_distance = mask_close_to_edges(mask, bboxes)
        if is_close is True:
            upd_image = np.zeros(shape=(image.shape[0], image.shape[1] + 2 * max_distance, image.shape[2]))
            upd_image[:, max_distance:-max_distance, :] = np.copy(image)
            upd_image[:, :max_distance ,:] = np.copy(image[:, -max_distance:, :])
            upd_image[:, -max_distance:, :] = np.copy(image[:, :max_distance, :])
            image = upd_image.astype(np.uint8)

            upd_mask = np.zeros(shape=(mask.shape[0], mask.shape[1] + 2 * max_distance))
            upd_mask[:, max_distance:-max_distance] = np.copy(mask)
            upd_mask[:, :max_distance] = np.copy(mask[:, -max_distance:])
            upd_mask[:, -max_distance:] = np.copy(mask[:, :max_distance])
            mask = upd_mask.astype(np.uint8)
    
            # adjust x coordinates of bboxes
            for i in range(len(bboxes)):
                bboxes[i] = (bboxes[i][0] + max_distance, *bboxes[i][1:])

        if PATCHING_TYPE == 'CONTOUR':                                                                                                                                                                                              
            # try stride max(w, h) // n
            stride = INPUT_SIZE // 4

            # loop bboxes with only one external contour
            for bidx, bbox in enumerate(bboxes):
                print(f'Box: {bidx}')
                t = time.time()
                x, y, w, h = bbox
                mask_bbox = mask[y:y+h, x:x+w]

                # while bbox of that contour has white pixels
                while cv2.findNonZero(mask_bbox) is not None: 
                    # find first contour
                    # TODO: after patching there could be multiple contours
                    contour = utils.get_contours(image=mask_bbox)[0]

                    # do patching while point idx is in range 
                    pidx = 0
                    while pidx < len(contour):
                        coord = contour[pidx]
                        py, px = coord[0][1], coord[0][0] 
                        print(f'Contour point: y{py} x{px}')

                        diff = INPUT_SIZE // 2
                        mask_patch = mask[y+py-diff:y+py+diff, x+px-diff:x+px+diff]
                        image_patch = image[y+py-diff:y+py+diff, x+px-diff:x+px+diff]

                        # check for white pixel
                        if cv2.findNonZero(mask_patch) is None:
                            pidx += stride
                            continue

                        # check shapes are correct
                        if image_patch[:,:,0].shape != (INPUT_SIZE, INPUT_SIZE) or mask_patch.shape != (INPUT_SIZE, INPUT_SIZE):
                            print('Incorrect patch size')
                            exit(-1)

                        # inference
                        inpaint_patch = static_inference(sess, input_layer, output_layer, image_patch, mask_patch)

                        # leave half of the mask to next patch to inpaint (residual)
                        mask_patch = mask_patch / 255.
                        residual_mask_patch = np.array(mask_patch, copy=True)
                        residual = stride // 2

                        # calculate from which side use residual
                        if pidx + stride < len(contour):
                            coord_next = contour[pidx + stride]
                            py_next, px_next = coord_next[0][1], coord_next[0][0]
                            if py_next - py > 0:
                                residual_mask_patch[-residual:, :] = 0
                            elif py_next - py < 0:
                                residual_mask_patch[:residual, :] = 0
                            if px_next - px > 0:
                                residual_mask_patch[:, -residual:] = 0
                            elif px_next - px < 0:
                                residual_mask_patch[:, :residual] = 0

                        # blend mask patch subtracted by residual into mask
                        mask[y+py-diff:y+py+diff, x+px-diff:x+px+diff] = ((mask_patch - residual_mask_patch) * 255.).astype(np.uint8)  
                        # blend inpaint patch into output patch
                        mask_patch = np.dstack([mask_patch] * 3)
                        residual_mask_patch = np.dstack([residual_mask_patch] * 3)
                        output_patch = inpaint_patch * residual_mask_patch + image_patch * (1. - residual_mask_patch)
                        output_patch = output_patch.astype(np.uint8)
                        # blend output patch into image
                        image[y+py-diff:y+py+diff, x+px-diff:x+px+diff] = output_patch

                        if LOCAL_CACHE is True:
                            cache(args.patch_dir, artifact_name, bidx, pidx, image_patch, mask_patch, inpaint_patch, output_patch)
                        pidx += stride
                print(f'Time on one bbox {bbox} inference: {time.time() - t}') 
        elif PATCHING_TYPE == 'CONV':
            # loop bboxes with only one external contour
            for bidx, bbox in enumerate(bboxes):
                print(f'Box: {bidx}')
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

                        # check shapes are correct
                        if image_patch[:,:,0].shape != (INPUT_SIZE, INPUT_SIZE) or mask_patch[:,:].shape != (INPUT_SIZE, INPUT_SIZE):
                            print('Incorrect patch size')
                            exit(-1)

                        # inference
                        inpaint_patch = static_inference(sess, input_layer, output_layer, image_patch, mask_patch)

                        # leave half of the hole to next patch to inpaint
                        mask_patch = mask_patch / 255.
                        partial_mask_patch = np.array(mask_patch, copy=True)
                        partial_mask_patch[:, -stride // 2:] = 0

                        # blend mask patch subtracted by residual into mask
                        mask[py:py+INPUT_SIZE, px:px+INPUT_SIZE] = ((mask_patch - partial_mask_patch) * 255.).astype(np.uint8)
                        # blend inpaint into output
                        mask_patch = np.dstack([mask_patch] * 3)
                        partial_mask_patch = np.dstack([partial_mask_patch] * 3)
                        output_patch = inpaint_patch * partial_mask_patch + image_patch * (1. - partial_mask_patch)
                        output_patch = output_patch.astype(np.uint8)
                        # blend output patch into image
                        image[py:py+INPUT_SIZE, px:px+INPUT_SIZE] = output_patch
                    
                        if LOCAL_CACHE is True:
                            cache(args.patch_dir, artifact_name, bidx, pidx, image_patch, mask_patch, inpaint_patch, output_patch)
                        
                        i += 1 
                        px += stride
                    px = x - INPUT_SIZE // 2
                    py += stride
                print(f'Time on one bbox {bbox} inference: {time.time() - t}') 
        elif PATCHING_TYPE == 'CONVEX':
            # try stride max(w, h) // n
            stride = INPUT_SIZE // 4

            # mask_copy = mask.copy()
            # mask_copy = np.dstack([mask_copy] * 3)
            # contours = utils.get_contours(image=mask)
            # hull = [cv2.convexHull(np.concatenate(tuple(contours), axis=0))]
            # cv2.drawContours(mask_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)
            # cv2.drawContours(mask_copy, contours=hull, contourIdx=-1, color=(0, 255, 0), thickness=8, lineType=cv2.LINE_AA)
            
            t = time.time()
            # while mask has white pixels
            while cv2.findNonZero(mask) is not None: 
                # find all contours
                contours = utils.get_contours(image=mask)
                hull = cv2.convexHull(np.concatenate(tuple(contours), axis=0))

                # do patching while point idx is in range 
                pidx = 0
                while pidx < len(hull):
                    coord = hull[pidx]
                    py, px = coord[0][1], coord[0][0] 
                    print(f'Hull point: y{py} x{px}')

                    diff = INPUT_SIZE // 2
                    mask_patch = mask[py-diff:py+diff, px-diff:px+diff]
                    image_patch = image[py-diff:py+diff, px-diff:px+diff]

                    # check for white pixel
                    if cv2.findNonZero(mask_patch) is None:
                        # pidx += stride
                        pidx += 1
                        continue

                    # check shapes are correct
                    if image_patch[:,:,0].shape != (INPUT_SIZE, INPUT_SIZE) or mask_patch.shape != (INPUT_SIZE, INPUT_SIZE):
                        print('Incorrect patch size')
                        exit(-1)

                    # inference
                    inpaint_patch = static_inference(sess, input_layer, output_layer, image_patch, mask_patch)

                    # # leave half of the mask to next patch to inpaint (residual)
                    # mask_patch = mask_patch / 255.
                    # residual_mask_patch = np.array(mask_patch, copy=True)
                    # residual = stride // 2

                    # # calculate from which side use residual
                    # if pidx + stride < len(contour):
                    #     coord_next = contour[pidx + stride]
                    #     py_next, px_next = coord_next[0][1], coord_next[0][0]
                    #     if py_next - py > 0:
                    #         residual_mask_patch[-residual:, :] = 0
                    #     elif py_next - py < 0:
                    #         residual_mask_patch[:residual, :] = 0
                    #     if px_next - px > 0:
                    #         residual_mask_patch[:, -residual:] = 0
                    #     elif px_next - px < 0:
                    #         residual_mask_patch[:, :residual] = 0

                    # mask[y+py-diff:y+py+diff, x+px-diff:x+px+diff] = ((mask_patch - residual_mask_patch) * 255.).astype(np.uint8)  
                    # blend inpaint patch into output patch
                    # residual_mask_patch = np.dstack([residual_mask_patch] * 3)
                    mask_patch = mask_patch / 255.
                    mask_patch = np.dstack([mask_patch] * 3)
                    output_patch = inpaint_patch * mask_patch + image_patch * (1. - mask_patch)
                    output_patch = output_patch.astype(np.uint8)
                    # blend output patch into image
                    image[py-diff:py+diff, px-diff:px+diff] = output_patch
                    # blend mask patch subtracted by residual into mask
                    mask[py-diff:py+diff, px-diff:px+diff] = 0

                    # if LOCAL_CACHE is True:
                    #     cache(args.patch_dir, artifact_name, bidx, pidx, image_patch, mask_patch, inpaint_patch, output_patch)
                    # pidx += stride
                # print(f'Time on one bbox {bbox} inference: {time.time() - t}')             

        # trim image back to save
        if is_close:
            image = image[:, max_distance:-max_distance, :]
            mask = mask[:, max_distance:-max_distance]
        filename = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path_image))[0] + INPAINT_SUFFIX)
        cv2.imwrite(filename, image)

if __name__ == "__main__":
    main()