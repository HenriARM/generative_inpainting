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


IMAGE_SUFFIX = '.jpg'


def multigpu_graph_def(model, FLAGS, data, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        images = data.data_pipeline(FLAGS.batch_size)
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = '/home/rudolfs/Desktop/trainings/training-pan-03-06-2021'
    # training data
    FLAGS = ng.Config('inpaint.yml')
    img_shapes = FLAGS.img_shapes
    datapaths = glob.glob(args.dataset + '/*' + IMAGE_SUFFIX)
    if len(datapaths) == 0:
        print('error')
        exit(-1)

    # shuffle data and split
    random.shuffle(datapaths)
    vlen = len(datapaths) // 4
    fnames, val_fnames = datapaths[:-vlen], datapaths[-vlen:]

    data = ng.data.DataFromFNames(
        fnames, img_shapes, queue_size=FLAGS.batch_size, enqueue_size=FLAGS.batch_size, random=True, random_crop=FLAGS.random_crop, nthreads=FLAGS.num_cpus_per_job)

    images = data.data_pipeline(FLAGS.batch_size)
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, images)

    # validation images
    if FLAGS.val:
        for i in range(vlen):
            static_fnames = [val_fnames[i]]
            static_images = ng.data.DataFromFNames(
                static_fnames, img_shapes, nthreads=1, queue_size=FLAGS.batch_size, enqueue_size=FLAGS.batch_size,
                random=True, random_crop=FLAGS.random_crop).data_pipeline(1)
            static_inpainted_images = model.build_static_infer_graph(
                FLAGS, static_images, name='static_view/%d' % i)


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
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
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
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
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

# TODO: tensorboard - batch incomplete is not inpainted, recheck
# TODO: tensorboard - update validation summary for epoch (seach in project "images_summary()")
# TODO: understand Hinge loses (gan_hinge_loss() in ./inpaint_model.py)
# TODO: understand Context Attention in ./inpaint_ops.py
# TODO: understand how kernel_spectral_norm in neuralgym/ops/gan_ops.py
# TODO: learn how to use graphs in Tensorboard
# TODO: store best loss, best epoch per each epoch (mean of all batches, not only last batch)
# TODO: add run.sh

# process mem size depends on batch_size and img_shapes - (batch 1 shape 256 - 16%, 550% ?%CPU, SHR, RES, VIRT)

if __name__ == "__main__":
    if tf.test.is_gpu_available():
        with tf.device('/gpu:0'):
            main()
    else:
        main()

"""
--logdir /home/rudolfs/Desktop/generative_inpainting/training --port 6006
ae_loss = L1 error of ground truth and coarse network + same of refine netwrok
"""

"""
Changes made to neuralgym:
TODO: added crop from center of pano 
Memory problems:
init_primary_trainer() in trainer.py
# TODO: 3 threads are in next_batch() and 2 batches are created in parallel
# TODO: add thread block when reading file and croping -> and releasing memory
# TODO: rewrite reading and croping image
# TODO: write separate script which reads panos and then check MEM size, is it released
from github issues, approx 0.5 sec for 1 batch default
"""
