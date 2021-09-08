import os
import glob

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

import neuralgymtf2 as ng
from inpaint_model import InpaintCAModel
# import utils
# import argparse
import random


FLAGS = ng.Config('inpaint.yml')
# ARTIFACT_CENTERED = True
# RANDOM_CROP = True
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
    val_len = 1 #len(file_paths) - 100 #1 #int(len(file_paths)* 0.2)
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