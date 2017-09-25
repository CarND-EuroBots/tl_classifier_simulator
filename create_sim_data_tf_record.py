"""Detect traffic ligths and their states in Udacity's
    self driving car simulator
"""

import hashlib
import io
import logging
import os
import random
import re
import yaml

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('input_image_width', 800, 'Width of sample images.')
flags.DEFINE_string('input_image_height', 600, 'Height of sample images.')
flags.DEFINE_string('sim_data_path', 'sim_data_large.yaml',
                    'Path to traffic light labels')

FLAGS = flags.FLAGS

def parse_yaml(sim_data_path):
  with open(sim_data_path, 'r') as stream:
    data_loaded = yaml.load(stream)
    return data_loaded



def dict_to_tf_example(example):
  """Convert YAML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(example['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = FLAGS.input_image_width
  height = FLAGS.input_image_height

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  for annotation in example['annotations']:
    xmin.append(float(annotation['xmin']) / width)
    ymin.append(float(annotation['ymin']) / height)
    xmax.append((float(annotation['xmin']) + float(annotation['x_width'])) / width)
    ymax.append((float(annotation['ymin']) + float(annotation['y_height'])) / height)
    class_name = annotation['class']
    classes_text.append(class_name.encode('utf8'))
    if(class_name == "Green"):
      class_label = 1
    elif(class_name == "Yellow"):
      class_label = 2
    elif(class_name == "Red"):
      class_label = 3
    classes.append(class_label)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          example['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          example['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def create_tf_record(sim_samples, output_filename):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(sim_samples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(sim_samples))
    path = os.path.join(example['filename'])

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue

    tf_example = dict_to_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
  sim_data_path = FLAGS.sim_data_path

  examples_list = parse_yaml(sim_data_path)

  # create_tf_record(examples_list, 'output')

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join('sim_train.record')
  val_output_path = os.path.join('sim_val.record')
  create_tf_record(train_examples, train_output_path)
  create_tf_record(val_examples, val_output_path)

if __name__ == '__main__':
  tf.app.run()
