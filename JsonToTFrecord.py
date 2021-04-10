import tensorflow as tf
import argparse
import json
import sys
import os.path
import pathlib
from pathlib import Path
from object_detection.utils import dataset_util, label_map_util
import io
import PIL
import hashlib
import random

parser = argparse.ArgumentParser()
parser.add_argument('--ImgInputDir', type = pathlib.Path)
parser.add_argument('--JsonFile', type = pathlib.Path)
parser.add_argument('--OutputDir', type = pathlib.Path)
parser.add_argument('--OutName', type = str)
parser.add_argument('--CategoryName', type = str)
parser.add_argument('--train_val_ratio', type = float)
args = parser.parse_args()

"""For my own benefit, here is what I'm pasting into the command line, change for whatever you need for the code
python JsonToTFrecord.py --ImgInputDir ./ImageFolders/TeamsData1 --JsonFile ./JsonFiles/TeamsJson_json.json --OutputDir ./TFRecords --OutName TeamsData1 --CategoryName Character --train_val_ratio .7
"""

labelmap = label_map_util.load_labelmap('./CharLabelMap.pbtxt')
labelmap = label_map_util.get_label_map_dict(labelmap)

def __main__():
    "You need to open AND load the json file"
    json_file = open(args.JsonFile)
    jsondict = json.load(json_file)

    keys = list(jsondict.keys())
    training_size = int(args.train_val_ratio*len(keys))
    random.shuffle(keys)

    traindata = list(keys)[:training_size]
    valdata = list(keys)[training_size:]

    dataset = [('_train', traindata), ('_val', valdata)]

    for suffix, data in dataset:
        output = os.path.join(args.OutputDir, args.OutName + suffix + ".tfrecord")
        with tf.io.TFRecordWriter(output) as writer:

            for index, key in enumerate(data): #Key is the raw image name and index is just the number
                try:
                    i = 1
                    for bbox in jsondict[key]['regions']:
                        print(f"Writing {jsondict[key]['filename']} bounding box #{i} to {output}")
                        i += 1
                        tf_example = dict_to_tf_example(jsondict[key], output, bbox)
                        writer.write(tf_example.SerializeToString())
                except UnboundLocalError as e:
                    if len(jsondict[key]['regions']) == 0:
                        print(f"Bounding error for {jsondict[key]['filename']}")
                    else:
                        print(e)

                except KeyError as f:
                    for i in jsondict[key]['regions']:
                        if i['region_attributes'] == 0:
                            print(f"Bounding error for {jsondict[key]['filename']}")
                        else:
                            print(f)

    print("Finish!")


def dict_to_tf_example(file, output, bounding_box):
    filename = file['filename']
    full_path = os.path.join(args.ImgInputDir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    size = file['size']
    prev_line = None
    shape = bounding_box['shape_attributes']

    image_width, image_height = image.size
    image_width = int(image_width)
    image_height = int(image_height)

    region_x = shape['x']
    region_y = shape['y']
    region_xmax = region_x + shape['width']
    region_ymax = region_y + shape['height']

    xmin.append(region_x / image_width)
    ymin.append(region_y / image_height)
    xmax.append(region_xmax / image_width)
    ymax.append(region_ymax / image_height)

    region_attributes = bounding_box['region_attributes']
    for k in region_attributes:
         category_name = region_attributes[k]

    classes_text.append(region_attributes[args.CategoryName].encode('utf8'))
    classes.append(labelmap[region_attributes[args.CategoryName]])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(image_height),
      'image/width': dataset_util.int64_feature(image_width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
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
    return tf_example


if __name__ == '__main__':
    __main__()

"""
Change
this
to
test
the
tfrecord"""
print(sum(1 for _ in tf.data.TFRecordDataset("./TFRecords/MangoArmadaMore_val.tfrecord")))
print(sum(1 for _ in tf.data.TFRecordDataset("./TFRecords/MangoArmadaMore_train.tfrecord")))
