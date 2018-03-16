"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.
Usage:
```shell
python tf_convert_dicom_to_tfrecord.py \
    --dataset_name=spine_segmentation \
    --dataset_dir=./datasets/spine_segmentation/spine_segmentation_5/test/ \
    --output_name=spine_segmentation_test_5_fold \
    --output_dir=./datasets/tfrecords_spine_segmentation_with_superpixels
```
"""
import tensorflow as tf

from datasets import convert_dicom_to_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'spine_segmentation_test_7_class',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'spine_segmentation':
        convert_dicom_to_tfrecord.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)
        
        
if __name__ == '__main__':
    tf.app.run()