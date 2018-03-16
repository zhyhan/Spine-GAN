"""Implementation of tf.metrics module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops


def create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
    """Creates a new local variable.
    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      collections: A list of collection names to which the Variable will be added.
      validate_shape: Whether to validate the shape of the variable.
      dtype: Data type of the variables.
    Returns:
      The created variable.
    """
    # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
    collections = list(collections or [])
    collections += [ops.GraphKeys.LOCAL_VARIABLES]
    return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)


def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.
  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.
  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  # Local variable to accumulate the predictions in the confusion matrix.
    cm_dtype = dtypes.int64 if weights is not None else dtypes.float64
    total_cm = create_local(
      'total_confusion_matrix',
      shape=[num_classes, num_classes],
      dtype=cm_dtype)

  # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)

  # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

  # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
          labels, predictions, num_classes, weights=weights, dtype=cm_dtype)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op        




def mean_Dice_score(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  
  with variable_scope.variable_scope(name, 'mean_Dice_score', (predictions, labels, weights)):
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total_cm, update_op = streaming_confusion_matrix(labels, predictions,
                                                      num_classes, weights)
    
    def compute_mean_Dice_score(name):
        _, total_cm_new = tf.split(total_cm,[1,6],1)
        _, total_cm_new = tf.split(total_cm_new,[1,6],0)
        print (total_cm.shape)
        """Compute the mean_Dice_score via the confusion matrix."""
        sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm_new, 0))
        sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm_new, 1))
        cm_diag = math_ops.to_float(array_ops.diag_part(total_cm_new))
        denominator = sum_over_row + sum_over_col

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
        denominator = array_ops.where(
            math_ops.greater(denominator, 0),
            denominator,
            array_ops.ones_like(denominator))
        Dice_score = math_ops.div(cm_diag*2, denominator)
        SS = math_ops.div(cm_diag, sum_over_col)
      #return math_ops.reduce_mean(Dice_score, name=name)
        return Dice_score, SS #return per class dsc
    mean_Dice_score_v, mean_SS = compute_mean_Dice_score('mean_Dice_score')


    return mean_Dice_score_v, mean_SS, update_op