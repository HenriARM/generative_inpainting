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
import sys
from scipy.linalg import sqrtm 

import numpy as np
from cv2 import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch

import neuralgymtf2 as ng
FLAGS = ng.Config('inpaint.yml')
from inpaint_model import InpaintCAModel

tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

INPUT_SIZE = 256
IMAGE_SUFFIX = '.jpg'
MASK_SUFFIX = '_mask.png'

########################################## MASK GENERATION ####################################################

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

def generate_mask():
    bbox = random_bbox(FLAGS)
    regular_mask = generate_regular_mask(FLAGS, bbox)
    irregular_mask = generate_irregular_mask(FLAGS)
    return np.logical_or(
        regular_mask.astype(np.bool),
        irregular_mask.astype(np.bool)).astype(np.float32)

############################################## FID ####################################################

# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def compile_fid(batch_size):
    image_ph = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, INPUT_SIZE, INPUT_SIZE, 3))
    model_input_size = min(INPUT_SIZE, 299)
    model_input_shape = (model_input_size, model_input_size, 3)
    # reshape image to input into model
    res_ph = tf.reshape(tensor=image_ph, shape=(image_ph.shape[0], *model_input_shape))
    # scale pixels between -1 and 1, sample-wise. 
    sc_ph = tf.compat.v1.keras.applications.inception_v3.preprocess_input(res_ph)
    model =  tf.compat.v1.keras.applications.inception_v3.InceptionV3(include_top=False,
        pooling='avg', input_shape=model_input_shape, weights='imagenet')
    return image_ph, model(sc_ph)

# input images of shape (B, H, W, C)
def calc_fid(input_layer, output_layer, img1, img2):
    # https://stackoverflow.com/questions/51107527/integrating-keras-model-into-tensorflow
    import tensorflow.compat.v1.keras.backend as K
    with K.get_session() as sess:
        K.set_session(sess)
        z1 = sess.run(output_layer, feed_dict={input_layer: img1})
        z2 = sess.run(output_layer, feed_dict={input_layer: img2})
          
    # calculate mean and covariance statistics (mean for all batches)
    mu1, sigma1 = z1.mean(axis=0), np.cov(z1, rowvar=False)
    mu2, sigma2 = z2.mean(axis=0), np.cov(z2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
	# calculate score
    fid_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_score.astype(np.float32)

################################################## INPAINTING #####################################################

# Builds model Graph
def compile_inpaint(checkpoint_dirname):      
    g = tf.compat.v1.Graph()
    with g.as_default():
        # init model graph
        model = InpaintCAModel()
        input_image_ph = tf.compat.v1.placeholder(tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE * 2, 3))
        output = model.build_server_graph(FLAGS, input_image_ph, reuse=tf.compat.v1.AUTO_REUSE)
        # post process graph
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # graph for assigning checkpoint loaded vars
        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            var_value = tf.train.load_variable(checkpoint_dirname, var.name)
            assign_ops.append(tf.compat.v1.assign(var, var_value))

    # create session with config for model and graph
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config, graph=g)
    # assign checkpoint vars
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

################################################## SUMMARY #####################################################

# TODO: TB in Runs written './', subfold each checkpoint https://github.com/tensorflow/tensorflow/issues/1548 name=f'step_{checkpoint_step}'
# TODO: tf.print is not working
# TODO: change each TB event file name, to know which checkpoint it is

# Builds Tensorboard summary Graph
def compile_val_summary(summary_dirname, checkpoint_step, batch_size):
    g = tf.compat.v1.Graph()
    with g.as_default():
        image_ph = tf.compat.v1.placeholder(tf.uint8, shape=(batch_size, INPUT_SIZE, INPUT_SIZE, 3))
        mask_ph = tf.compat.v1.placeholder(tf.uint8, shape=(batch_size, INPUT_SIZE, INPUT_SIZE, 3))
        output_ph = tf.compat.v1.placeholder(tf.uint8, shape=(batch_size, INPUT_SIZE, INPUT_SIZE, 3))
        fid_ph = tf.compat.v1.placeholder(tf.float32, shape=())

        # graph for creating Tensorboard file writer
        writer = tf.summary.create_file_writer(logdir=summary_dirname)
        with writer.as_default():
            # draw image, mask, output
            data = tf.concat([image_ph, mask_ph, output_ph], axis=2)
            tf.summary.image(name='val_inpaint', max_outputs=batch_size, data=data, step=checkpoint_step)

            # scalars
            l1 = tf.reduce_mean(tf.abs(tf.cast((image_ph - output_ph),dtype=tf.float32)))
            psnr = tf.reduce_mean(tf.image.psnr(image_ph, output_ph, max_val=255), axis=0)
            ssim = tf.reduce_mean(tf.image.ssim(image_ph, output_ph, max_val=255), axis=0)
            # ssim for all color channels
            ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(image_ph, output_ph, max_val=255), axis=0)
            
            tf.print(f'Metrics1:', output_stream=sys.stdout)
            tf.print(f'Metrics2: l1 {l1}, psnr {psnr}, ssim {ssim}, ms_ssim {ms_ssim}, fid {fid_ph}')
            tf.print(f'Metrics3: l1 {l1}, psnr {psnr}, ssim {ssim}, ms_ssim {ms_ssim}, fid {fid_ph}', output_stream=sys.stdout)

            tf.summary.scalar('l1', l1, checkpoint_step)
            tf.summary.scalar('psnr', psnr, checkpoint_step)
            tf.summary.scalar('ssim', ssim, checkpoint_step)
            tf.summary.scalar('ms_ssim', ms_ssim, checkpoint_step)
            # TODO: ?
            tf.summary.scalar('fid', fid_ph, checkpoint_step)

        summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        writer_flush = writer.flush()
    sess = tf.compat.v1.Session(graph=g)
    # init file_writer so during validation it will be available 
    sess.run(writer.init())
    print('Summary loaded')
    return sess, summary_ops, writer_flush, image_ph, mask_ph, output_ph, fid_ph

################################################## MAIN #####################################################

def np_concat(arr, el):
    el = np.expand_dims(el, axis=0)
    if arr is None:
        arr = el
    else:
        arr = np.concatenate((arr, el), axis=0)
    return arr

# TODO: Compile only once deepfill and inception, and summary graph 
def validate_checkpoint(checkpoint_dirname, checkpoint_step, image_abspaths, mask_abspaths):
    # compile DeepFill inpainting model
    model_sess, input_layer, output_layer = compile_inpaint(checkpoint_dirname=checkpoint_dirname)
    print('DeepFill compiled...')

    # run validation
    images = None
    masks = None
    outputs = None
    for image_abspath, mask_abspath in zip(image_abspaths, mask_abspaths):
        print(os.path.basename(image_abspath))

        image = cv2.imread(image_abspath)
        raw_mask = cv2.imread(mask_abspath)

        # convert mask to grayscale and threshold
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)

        # inference
        inpaint = static_inference(model_sess, input_layer, output_layer, image, mask)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.dstack([mask] * 3)
        mask_norm = mask / 255.
        output = inpaint * mask_norm + image * (1. - mask_norm)
        output = output.astype(np.uint8)
    
        # store for summary
        images = np_concat(images, image)
        masks = np_concat(masks, mask)
        outputs = np_concat(outputs, output)

        #cv2.imshow('img', image)
        #cv2.imshow('mask', mask)
        #cv2.imshow('inpaint', inpaint)
        #cv2.imshow('out', output)
        #cv2.waitKey(0)

    batch_size = images.shape[0]
    
    # compile Inception model for FID
    fid_input_layer, fid_output_layer = compile_fid(batch_size)
    print('InceptionV3 compiled...')
    fid_score = calc_fid(fid_input_layer, fid_output_layer, images, masks)
    print(f'FID score {fid_score}')

    # compile summary
    # TODO: change name of val_logs
    summary_sess, summary_ops, writer_flush, image_ph, mask_ph, output_ph, fid_ph = compile_val_summary(
        summary_dirname='/home/henri/projects/deepfill/tb', 
        checkpoint_step=checkpoint_step,
        batch_size=batch_size)

    # summary
    summary_sess.run(summary_ops, feed_dict={image_ph: images, mask_ph: masks, output_ph: outputs, fid_ph:fid_score})
    summary_sess.run(writer_flush)

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
            validate_checkpoint(checkpoint_dirname=checkpoint_dirname,
                checkpoint_step=last_snapshot,
                image_abspaths=image_abspaths,
                mask_abspaths=mask_abspaths)
        print('Throttle...')
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()

'''
FID calculation inside tf.Session using tf functions

def tf_cov(x):
    vx = tf.matmul(tf.transpose(x), x) / x.shape[0]
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean), mean)
    return tf.cast(vx - mx, tf.float32)

# input Z vector with (B, num_features)
def tf_mean_var(x):
    # feature-wise mean of the real and generated images, 
    # where each element is the mean feature observed across images
    mu = tf.reduce_mean(x, axis=0)
    sigma = tf_cov(x)
    return mu, sigma

# calculate sum squared difference between means
ssdiff = tf.reduce_sum((mu1 - mu2)**2.0)
# calculate sqrt of product between cov
covmean = tf.linalg.sqrtm(tf.tensordot(sigma1, sigma2, axes=1))
# calculate score
return ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2.0 * covmean)

'''