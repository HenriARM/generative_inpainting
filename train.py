import os
import glob

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

import neuralgymtf2 as ng
from inpaint_model import InpaintCAModel
import utils
import argparse
import random


FLAGS = ng.Config('inpaint.yml')
ARTIFACT_CENTERED = True
RANDOM_CROP = True
IMAGE_SHAPE = FLAGS.img_shape
DATASET_PATH = FLAGS.dataset_path
IMAGE_SUFFIX = '.jpg'


def multigpu_graph_def(model, FLAGS, images, gpu_id=0, loss_type='g'):
    #with tf.device('/cpu:0'):
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, images, FLAGS, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


def main():
    file_paths = glob.glob(FLAGS.dataset_path + '/*' + IMAGE_SUFFIX)
    if len(file_paths) == 0:
        print('error')
        exit(-1)

    # split data
    random.shuffle(file_paths)
    val_len = len(file_paths) - 100 #1 #int(len(file_paths)* 0.2)
    train_paths, val_paths = file_paths[:-val_len], file_paths[-val_len:]

    data = ng.data.DataFromFNames(
        train_paths, FLAGS.img_shape, queue_size=FLAGS.batch_size, enqueue_size=FLAGS.batch_size, random=True, random_crop=FLAGS.random_crop, nthreads=FLAGS.num_cpus_per_job)
 
    images = data.data_pipeline(FLAGS.batch_size)
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images)

    # training settings
    lr = tf.compat.v1.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.compat.v1.constant_initializer(1e-4))
    d_optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer

    # train discriminator with secondary trainer, should initialize before primary trainer.
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=1,
        grads_summary=False,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'images': images, 'loss_type': 'd'},
    )

    # train generator with primary trainer
    trainer = ng.train.Trainer(
        num_gpus=FLAGS.num_gpus_per_job,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=FLAGS.max_iters,
        graph_def=multigpu_graph_def,
        grads_summary=False,
        gradient_processor=None,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'images': images, 'loss_type': 'g'},
        spe=FLAGS.train_spe,
        log_dir=FLAGS.log_dir,
    )

    # add all callbacks
    trainer.add_callbacks([
        discriminator_training_callback,
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.model_restore+'/snap'), 
        ng.callbacks.SummaryWriter((FLAGS.val_psteps), trainer.context['summary_writer'], tf.compat.v1.summary.merge_all()),
    ])
    # launch training
    trainer.train()


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        main()
    # if tf.test.is_gpu_available():
    #     with tf.device('/gpu:0'):
    #         main()
    # else:
    #     main()

"""
--logdir /home/rudolfs/Desktop/generative_inpainting/training --port 6006
ae_loss = L1 error of ground truth and coarse network + same of refine netwrok
"""

# Options for net:
# Eager execution: tf.compat.v1.Layers + tf.compat.v1.Session()
# Static execution: tf.function + SavedModel pb

# # TensorFlow 1.X
# outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# # TensorFlow 2.0
# outputs = f(input)

# Read Fully QueueRunning and how threading is done
# 1. set tf log = true, to show cpu
# 0. how many threads we create? (besides for preprocessing)
# 2. if mem/time problem is with croping - make daudz cropus un uztrenet modeli
# 3. play with hardcoding of tf.device('/cpu:0') in preprocessing 


# TODO: maybe PIL and Wand will crop raw images (bmp, .tiff) without opening it? 
# find way how to use memory profiler inside of crop functions, maybe they read it whole in RAM


# TODO: terminate called without an active exception (probably problem with threads, not detaching or joining)
# TODO: put breakpoint on next_batch(), there is separate thread for running _run of QueueRunner. Trainer.py feed_dict=self.context['feed_dict']) is empty
# TODO: find other deepfill tf impl + which run fast with no gpu bottleneck and memory leak + see if neuralgym is used
# TODO: find all places where next_batch() is used (is it in trainer.py or not?)
# TODO: dont even need QueueRunner - for building graph use batch_ph, 
# TODO: ask Eduards how correctly should be visualised images in tensorboard?
# TODO: take 100 static images and show each epoch how they are inpainted?
# TODO: val_psteps == train_spe 
# TODO: train on server from local files, not /mnt
# TODO: tensorboard - update validation summary for epoch (seach in project "images_summary()")
# TODO: understand Hinge loses (gan_hinge_loss() in ./inpaint_model.py)
# TODO: understand Context Attention in ./inpaint_ops.py
# TODO: understand how kernel_spectral_norm in neuralgym/ops/gan_ops.py
# TODO: learn how to use graphs in Tensorboard
# TODO: store best loss, best epoch per each epoch (mean of all batches, not only last batch)
# TODO: add run.sh

# static_images = ng.data.DataFromFNames().data_pipeline(1)
# static_inpainted_images = model.build_static_infer_graph(FLAGS, images, name='static_view/%d' % i)

'''
train_dataset = ng.data.data_from_fnames.create_dataset(file_paths=train_paths,batch_size=FLAGS.batch_size)
dataset_iter = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
x = dataset_iter.get_next()

batch_phs = tf.compat.v1.placeholder(tf.float32, [FLAGS.batch_size] + FLAGS.img_shape)
model = InpaintCAModel()
g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, batch_phs)

sess = tf.compat.v1.Session()
one = np.ones(shape=(16,256,256,3), dtype=np.float32)
res = sess.run(batch_phs, feed_dict={batch_phs: one})
for op in sess.graph.get_operations():
    print(str(op.name))

Results of dequeueing:
Placeholder
FIFOQueueV2
FIFOQueueV2_EnqueueMany
FIFOQueueV2_Close
FIFOQueueV2_Close_1
FIFOQueueV2_DequeueMany/n
FIFOQueueV2_DequeueMany
'''

'''
Results:
-------
4 threads, 8 batch size
Wand - 30 and exited, 25% mem with increasing to 700% CPU
OpenCV - 4 and exited, 70% mem used and 600% CPU
PIL - 35 and exited, 70% mem and till 700% CPU
-------
1 threads, 8 batch size
Wand - 35 and exited, 30% mem and till 700% CPU
OpenCV - 35 and exited, 40% mem and till 700% CPU
PIL - 35 and exited, 30% mem and till 700% CPU
-------
Conclusion: num_threads will imply on amount of FIFO pipelines 
=> only mem % will increase with increasing num of threads
'''