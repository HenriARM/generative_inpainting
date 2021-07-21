import os
import tensorflow as tf
from absl import flags
from absl import app
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "/home/henri/projects/deepfill/models/mytrainpb/saved_model.pb", "tensorflow frozen model")
flags.DEFINE_string("output_dir", "./tmp", "where to save the slutty reslut")
flags.DEFINE_boolean("add_jpeg_input", False, "Add jpeg decode ops at the beginning of model graph, so that jpeg can be sent over API insted of float array")
flags.DEFINE_boolean("add_extra_input_channel", False, "Add extra channel for for depth map or histogram projection")
flags.DEFINE_string("signature_def_map", 'deeplab', "signature map of tensorflow serving model")

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'heatmaps:0'


def main(argv):
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  # graph = tf.Graph()

  path = tf.keras.utils.get_file(
    'inception_v1_2016_08_28_frozen.pb',
    'http://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz',
    untar=True)

  graph_def = tf.compat.v1.GraphDef()
  loaded = graph_def.ParseFromString(open(FLAGS.model_path,'rb').read())


  if graph_def is None:
    raise RuntimeError('Model not found.')

  with graph.as_default():
    if FLAGS.add_jpeg_input:
      jpeg_batch = tf.compat.v1.placeholder(tf.string, name="encoded_input", shape=[1])
      images_tensor = tf.map_fn(lambda image: tf.cast(tf.io.decode_image(image), tf.dtypes.float32) * (2.0 / 255.0) - 1.0, jpeg_batch, dtype=tf.float32)
      if FLAGS.add_extra_input_channel:
          png_batch = tf.compat.v1.placeholder(tf.string, name="encoded_extra_input", shape=[1])
          images_extra_tensor = tf.map_fn(lambda image: tf.cast(tf.io.decode_image(image), tf.dtypes.float32) * (2.0 / 255.0) - 1.0, png_batch, dtype=tf.float32)
          images_tensor = tf.concat([images_tensor, images_extra_tensor], axis=3)

      tf.import_graph_def(graph_def, name='', input_map={INPUT_TENSOR_NAME:images_tensor})
    else:
      tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session(graph=graph) as sess:
      print('Model loaded successfully!')

      shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
      os.makedirs(FLAGS.output_dir, exist_ok=True)

      tensor_info_input = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name("encoded_input:0" if FLAGS.add_jpeg_input else INPUT_TENSOR_NAME))
      if FLAGS.add_extra_input_channel:
        tensor_info_extra_input = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name("encoded_extra_input:0"))

      tensor_info_output = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name(OUTPUT_TENSOR_NAME))

      inputs = {'images': tensor_info_input}
      if FLAGS.add_extra_input_channel:
          inputs["images_extra_channel"] = tensor_info_extra_input
 

      prediction_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs=inputs,
          outputs={'class_heatmaps': tensor_info_output},
          method_name=tf.saved_model.PREDICT_METHOD_NAME))

      builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(FLAGS.output_dir)

      builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.SERVING],
        signature_def_map={
          FLAGS.signature_def_map:
          prediction_signature })

      builder.save(as_text=True)
      print('Done exporting!')



if __name__ == '__main__':
  app.run(main)
