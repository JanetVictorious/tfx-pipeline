import tensorflow_transform as tft

import helper_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = helper_constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = helper_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = helper_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = helper_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = helper_constants.LABEL_KEY
_transformed_name = helper_constants.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_by_min_max(
            inputs[key])

    # Bucketize these features
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

    # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key])

    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[_LABEL_KEY])

    return outputs
