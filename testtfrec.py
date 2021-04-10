import tensorflow.compat.v1 as tf

"""THIS WORKS but requires me to use tensor flow version 1"""

for example in tf.python_io.tf_record_iterator("./TFRecords/MangoArmadaMore_train.tfrecord"):
    result = tf.train.Example.FromString(example)
print(result)
