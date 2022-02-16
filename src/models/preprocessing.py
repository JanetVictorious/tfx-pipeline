"""TFX taxi preprocessing.

This file defines a template for TFX Transform component and uses features
defined in features.py.
"""

from typing import Optional

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_probability as tfp

from models import features


def _fill_in_missing(x: tf.sparse.SparseTensor,
                     fill_value=None):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with either specified `fill_strategy`,
    `fill_value`, or '' or 0, and converts to a dense tensor.

    :param SparseTensor x:
        Of rank 2.  Its dense shape should have size at most 1 in the
        second dimension.
    :param fill_value:
        Specified value to impute missing values.

    :return:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    # if fill_strategy == 'mean':
    #     default_value = tft.mean(x)
    # elif fill_strategy == 'median':
    #     default_value = tfp.stats.percentile(
    #         x, 50.0, interpolation='midpoint')
    #     # if x.dtype in [tf.int32, tf.int64]:
    #     #     default_value = tfp.stats.percentile(
    #     #         x, 50.0, interpolation='nearest')
    #     # else:
    #     #     default_value = tfp.stats.percentile(
    #     #         x, 50.0, interpolation='midpoint')
    if fill_value is not None:
        default_value = fill_value
    else:
        default_value = '' if x.dtype == tf.string else 0

    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


# TODO: Implement configuration for transform output
# ref: https://www.tensorflow.org/tfx/guide/transform


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    :param inputs:
      Map from feature keys to raw not-yet-transformed features.
    :return:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    for key in features.DENSE_FLOAT_FEATURE_KEYS:
        # NOTE: Impute with mean/median for float features
        # OPTION 1: MEAN
        # avg = tft.mean(inputs[key])
        # outputs[features.transformed_name(key)] = tft.scale_to_z_score(
        #     _fill_in_missing(inputs[key], fill_value=avg))
        # OPTION 2: MEDIAN (requires tensorflow_probability)
        # REFERENCE: https://stackoverflow.com/questions/43824665/tensorflow-median-value
        # med = tfp.stats.percentile(inputs[key], 50.0, interpolation='midpoint')
        # outputs[features.transformed_name(key)] = tft.scale_to_z_score(
        #     _fill_in_missing(inputs[key], fill_value=med))

        # If sparse make it dense, impute missing values, and apply zscore.
        outputs[features.transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key],
                             fill_value=tf.cast(tft.mean(inputs[key]),
                                                inputs[key].dtype)))

    for key in features.VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[features.transformed_name(key)] = tft\
          .compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=features.VOCAB_SIZE,
            num_oov_buckets=features.OOV_SIZE)

    # for key, num_buckets in zip(features.BUCKET_FEATURE_KEYS,
    #                             features.BUCKET_FEATURE_BUCKET_COUNT):
    for key, num_buckets in features.BUCKET_FEATURE_DICT.items():
        outputs[features.transformed_name(key)] = tft.bucketize(
            _fill_in_missing(inputs[key]),
            num_buckets)

    for key in features.CATEGORICAL_FEATURE_KEYS:
        outputs[features.transformed_name(key)] = _fill_in_missing(inputs[key])

    # # NOTE: One-Hot encoding strategy
    # # Convert strings to indices and convert to one-hot vectors
    # for key, vocab_size in features.VOCAB_FEATURE_DICT.items():
    #     indices = tft.compute_and_apply_vocabulary(
    #         inputs[key], num_oov_buckets=features.OOV_SIZE)
    #     one_hot = tf.one_hot(indices, vocab_size + features.OOV_SIZE)
    #     outputs[features.transformed_name(key)] = tf.reshape(
    #         one_hot, [-1, vocab_size + features.OOV_SIZE])

    # # Bucketize this feature and convert to one-hot vectors
    # for key, num_buckets in _BUCKET_FEATURE_DICT.items():
    #     indices = tft.bucketize(inputs[key], num_buckets)
    #     one_hot = tf.one_hot(indices, num_buckets)
    #     outputs[key] = tf.reshape(one_hot, [-1, num_buckets])

    # Was this passenger a big tipper?
    taxi_fare = _fill_in_missing(inputs[features.FARE_KEY])
    tips = _fill_in_missing(inputs[features.LABEL_KEY])
    outputs[features.transformed_name(features.LABEL_KEY)] = tf.where(
        tf.math.is_nan(taxi_fare),
        tf.cast(tf.zeros_like(taxi_fare), tf.int64),
        # Test if the tip was > 20% of the fare.
        tf.cast(tf.greater(
          tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

    return outputs
