'''
Validation script. Run during training in parallel to validate with static images and generated masks
last training checkpoints.
Generate dataset of images before using this script.
'''
import time
import os
import glob
import math
import argparse

import numpy as np
from cv2 import cv2
from PIL import Image, ImageDraw
import tensorflow as tf

import neuralgymtf2 as ng
FLAGS = ng.Config('inpaint.yml')
from inpaint_model import InpaintCAModel

tf.compat.v1.disable_eager_execution()

INPUT_SIZE = 256
IMAGE_SUFFIX = '.jpg'
MASK_SUFFIX = '_mask.png'

def random_bbox(FLAGS):
    # returns tuple (top, left, height, width)
    maxt = FLAGS.img_shapes[0] - FLAGS.vertical_margin - FLAGS.height
    maxl = FLAGS.img_shapes[1] - FLAGS.horizontal_margin - FLAGS.width
    t = int(np.random.uniform(low=FLAGS.vertical_margin, high=maxt))
    l = int(np.random.uniform(low=FLAGS.horizontal_margin, high=maxl))
    return t, l, FLAGS.height, FLAGS.width

def generate_regular_mask(FLAGS, bbox):
    # returns mask with shape [1, H, W, 1]
    bbox_t, bbox_l, bbox_h, bbox_w = bbox
    mask = np.zeros((1, FLAGS.img_shapes[0], FLAGS.img_shapes[1], 1), np.float32)
    h = np.random.randint(FLAGS.max_delta_height//2+1)
    w = np.random.randint(FLAGS.max_delta_width//2+1)
    mask[:, bbox_t:bbox_t+bbox_h, bbox_l+w:bbox_l+bbox_w-w, :] = 1.
    return mask

def generate_irregular_mask(FLAGS):
    # returns irregular mask with shape [1, H, W, 1]
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40

    img_shape = FLAGS.img_shapes
    height = img_shape[0]
    width = img_shape[1]
    H = height
    W = width

    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((
                v[0] - width//2,
                v[1] - width//2,
                v[0] + width//2,
                v[1] + width//2),
                fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, H, W, 1))
    return mask

def model_compile(checkpoint_dir):
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
        var_value = tf.train.load_variable(checkpoint_dir, from_name)
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

def generate_mask():
    bbox = random_bbox(FLAGS)
    regular_mask = generate_regular_mask(FLAGS, bbox)
    irregular_mask = generate_irregular_mask(FLAGS)
    return np.logical_or(
        regular_mask.astype(np.bool),
        irregular_mask.astype(np.bool)).astype(np.float32)

def validate(checkpoint_dir, image_abspaths, mask_abspaths):
    # compile model
    sess, input_layer, output_layer = model_compile(checkpoint_dir=checkpoint_dir)  

    # run validation
    for image_abspath, mask_abspath in zip(image_abspaths, mask_abspaths):
        print(os.path.basename(image_abspath))

        image = cv2.imread(image_abspath)
        raw_mask = cv2.imread(mask_abspath)

        # convert mask to grayscale and threshold
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)

        # inference
        inpaint = static_inference(sess, input_layer, output_layer, image, mask)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.dstack([mask] * 3)
        mask = mask / 255.
        output = inpaint * mask + image * (1. - mask)
        output = output.astype(np.uint8)
        #cv2.imshow('img', image)
        #cv2.imshow('mask', mask)
        #cv2.imshow('inpaint', inpaint)
        #cv2.imshow('out', output)
        #cv2.waitKey(0)


def get_current_snapshot(ckpt):
    f = open(ckpt, 'r')
    snapshot = f.readline()
    f.close()
    # e.x. 'model_checkpoint_path: "snap-1000"'
    return int(snapshot.replace('\"', '').split('-')[-1])
    
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.training_dir = '/home/henri/projects/deepfill/training'
    args.val_images_relpath = 'val_images'
    args.val_masks_relpath = 'val_masks'
    args.val_logs_relpath = 'val_logs'
    args.checkpoint_relpath = 'checkpoints'

    val_images_dirname = os.path.join(args.training_dir, args.val_images_relpath)
    val_masks_dirname = os.path.join(args.training_dir, args.val_masks_relpath)
    val_logs_dirname = os.path.join(args.training_dir, args.val_logs_relpath)
    checkpoint_dirname = os.path.join(args.training_dir, args.checkpoint_relpath)

    # check all folders exist
    if not os.path.exists(args.training_dir):
        print('Training rootpath doesnt exist')
        exit(-1)

    if not os.path.exists(val_images_dirname):
        print('Validation images dont exist')
        exit(-1)

    if not os.path.exists(checkpoint_dirname ):
        print('Training checkpoints dont exist')
        exit(-1)

    if not os.path.exists(val_logs_dirname):
        os.makedirs(val_logs_dirname)
        print('Validation logs didnt exist, created folder')

    masks_exist = True
    if not os.path.exists(val_masks_dirname):
        masks_exist = False
        os.makedirs(val_masks_dirname)
        print('Validation masks didnt exist, created folder')    

    # file from which we will read last trained checkpoint
    checkpoint_file_abspath = os.path.join(checkpoint_dirname, 'checkpoint')

    def sort(str_lst):
        return [s for s in sorted(str_lst)]

    # read all images
    image_abspaths = glob.glob(val_images_dirname + '/*' + IMAGE_SUFFIX)
    image_abspaths = sort(image_abspaths)

    # create unique mask for each crop (store locally to continue validation after stop)
    if masks_exist is False:
        for image_abspath in image_abspaths:
            mask = (generate_mask()[0] * 255).astype(np.uint8)
            cv2.imwrite(
                f'{image_abspath.replace(args.val_images_relpath, args.val_masks_relpath).replace(IMAGE_SUFFIX, MASK_SUFFIX)}', 
                mask)

    # read all masks
    mask_abspaths = glob.glob(val_masks_dirname + '/*' + MASK_SUFFIX)
    mask_abspaths = sort(mask_abspaths)

    # check same amount of images and masks
    assert len(image_abspaths) == len(mask_abspaths)

    # check each interval for new trained checkpoint
    INTERVAL = 5 # 5 * 60
    last_snapshot = -1
    while True:
        new_snapshot = get_current_snapshot(ckpt=checkpoint_file_abspath)
        if new_snapshot > last_snapshot:
            last_snapshot = new_snapshot
            print(f'New snapshot: {last_snapshot}')
            print('Valdiation...')
            validate(
                checkpoint_dir=checkpoint_dirname,
                image_abspaths=image_abspaths,
                mask_abspaths=mask_abspaths)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()