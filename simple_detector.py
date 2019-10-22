import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
import skvideo.io

import random
import string
import time
from shapely.geometry import Polygon  # Pacote que verifica a sobreposição de polígonos
import json
import scipy.misc


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/davi/Documentos/Seccomp/object_detection/models/research/object_detection/")
sys.path.append("/home/davi/Documentos/Seccomp/object_detection/models/research/")
from object_detection.utils import ops as utils_ops
print("Versão do tensorflow: ")
print(tf.__version__)


if tf.__version__ < '1.10.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
import cv2
import time


PATH_TO_CKPT = "/home/davi/Documentos/Seccomp/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("/home/davi/Documentos/Seccomp/object_detection/models/research/object_detection/", 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, sess, tensor_dict):


  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  startTime = time.time()
  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})
  print(time.time() - startTime)


  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def init():


	with detection_graph.as_default():    
		with tf.Session() as sess:

			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

			frame = cv2.imread("persons.jpeg")

			output_dict = run_inference_for_single_image(frame, sess, tensor_dict)

			vis_util.visualize_boxes_and_labels_on_image_array(
			    frame,
			    output_dict['detection_boxes'],
			    output_dict['detection_classes'],
			    output_dict['detection_scores'],
			    category_index,
			    instance_masks=output_dict.get('detection_masks'),
			    use_normalized_coordinates=True,
			    line_thickness=1
			    )
			
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			scipy.misc.imsave('outfile.jpg', frame)



if __name__ == '__main__':

  init() # Processar vídeo salvo
