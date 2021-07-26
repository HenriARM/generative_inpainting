from base64 import b64decode
import numpy as np
from cv2 import cv2
import os
import utils
import glob
import argparse
import neuralgymtf2 as ng

import tensorflow as tf
from inpaint_model import InpaintCAModel

tf.compat.v1.disable_eager_execution()

CHECKPOINT_DIR = '/home/henri/projects/deepfill/models/mytrain'
INPUT_SIZE = 256  # input image size for Generator 512
IMAGE_SUFFIX = '_hdrnet.jpg'
MASK_SUFFIX = '_inpainted_mask.png'
INPAINT_SUFFIX = '_inpainted.jpg'

MIN_BBOX_AREA = 50 * 50
OVERLAP_DISTANCE = 200

INPUT_TENSOR_NAME = 'encoded_input'
OUTPUT_TENSOR_NAME = 'saturate_cast'

# def inferencetf2(image, mask):
#     # prepare input tensor
#     mask = np.dstack([mask] * 3)
#     mask = np.expand_dims(mask, 0)
#     image = np.expand_dims(image, 0)
#     input_image = np.concatenate([image, mask], axis=2)
    
#     with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
#         tf.compat.v1.saved_model.loader.load(sess, ['serve'], './modeltf2')
#         graph = tf.compat.v1.get_default_graph()
#         input_tensor = graph.get_tensor_by_name('Placeholder:0')
#         output_tensor = graph.get_tensor_by_name('saturate_cast:0')
#         # inference
#         output_image = sess.run(output_tensor,feed_dict={input_tensor: input_image})
#         output_image = output_image[0][:, :, ::-1]
#         return output_image

# TODO: rewrite to send only image file_path
# TODO: rewrite for multiple images [for each image byte array -> decode to (1, 256, 512, 3)] 
def export_model():
    FLAGS = ng.Config('inpaint.yml')
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    graph = tf.compat.v1.get_default_graph()
    # graph_def = graph.as_graph_def()
      
    jpeg_batch = tf.compat.v1.placeholder(dtype=tf.string, name=INPUT_TENSOR_NAME, shape=[1])
    images_tensor = tf.map_fn(lambda image: tf.cast(tf.io.decode_image(image), tf.float32), jpeg_batch, dtype=tf.float32)
    images_tensor.set_shape(shape=(1, INPUT_SIZE, INPUT_SIZE * 2, 3))
    # image_tensor = tf.expand_dims(image_tensor, axis=0)
    # input_image_ph = tf.compat.v1.placeholder(name='Placetest', dtype=tf.float32, shape=(INPUT_SIZE, INPUT_SIZE * 2, 3))

    ############################# CODE DUPLICATE ################################
    model = InpaintCAModel()
    # input_image_ph = tf.compat.v1.placeholder(tf.float32, shape=(1, INPUT_SIZE, INPUT_SIZE * 2, 3))
    output = model.build_server_graph(FLAGS, images_tensor, reuse=tf.compat.v1.AUTO_REUSE)
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
    ############################# CODE DUPLICATE ################################

    tensor_info_input = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name(f'{INPUT_TENSOR_NAME}:0'))
    tensor_info_output = tf.compat.v1.saved_model.build_tensor_info(graph.get_tensor_by_name(f'{OUTPUT_TENSOR_NAME}:0')) #map/TensorArrayV2Stack/TensorListStack:0      
    inputs = {'images': tensor_info_input}
    outputs = {'class_heatmaps': tensor_info_output}
    prediction_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.PREDICT_METHOD_NAME
        )
    )
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./test')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.SERVING],
        signature_def_map={'deeplab':prediction_signature}
    )
    builder.save(as_text=True)
    print('Done exporting!')


def inference(image, mask):
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

    # # save model as protobuf
    # graph = sess.graph
    # input_tensor = graph.get_tensor_by_name('Placeholder:0')
    # output_tensor = graph.get_tensor_by_name('saturate_cast:0')
    # tf.compat.v1.saved_model.simple_save(
    #     sess,"./modeltf2",
    #     inputs={'Placeholder:0':input_tensor}, 
    #     outputs={'saturate_cast:0': output_tensor}
    # )
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

    args.dataset = '/home/henri/datasets/artifacts/panos/pan-21-07-2021/data'
    args.output_dir = './output'  # output directory

    paths_image, paths_mask = utils.read_paths(dataset_path=args.dataset, image_suffix=IMAGE_SUFFIX, mask_suffix=MASK_SUFFIX)

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
        bboxes, mask = utils.get_bboxes(contours=contours, mask=mask, min_bbox_area=MIN_BBOX_AREA, overlap_distance=OVERLAP_DISTANCE)

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

def test_serve():
    import requests
    import base64
    import json
    import numpy as np

    MODEL_BASE_PATH = 'models'
    MODEL_NAME = 'deeplab'
    url = f'http://localhost:8501/v1/{MODEL_BASE_PATH}/{MODEL_NAME}:predict'
    image_path = '/home/henri/Downloads/merged.jpg'

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    data = {"signature_name": "deeplab", "inputs": {"images": [{"b64": base64.b64encode(image_bytes).decode("utf-8")}]}}
    response = requests.post(url, data=json.dumps(data))
    if response.status_code == 200:
        image = json.loads(response.text)['outputs']
        image = np.asarray(image, dtype=np.uint8)
        cv2.imwrite('./img.jpg', image[0])


if __name__ == "__main__":
    # export_model()
    # test_serve
    main()