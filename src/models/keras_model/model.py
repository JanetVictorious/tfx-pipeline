"""TFX template taxi model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

from absl import logging
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from models import features, preprocessing
from models.keras_model import constants

# TFX Transform will call this function.
preprocessing_fn = preprocessing.preprocessing_fn

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def _get_hyperparameters() -> kt.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = kt.HyperParameters()

    # Defines search space.
    hp.Choice(name='learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)
    # hp.Float(name='learning_rate', min_value=1e-4,
    #          max_value=1e-1, default=1e-3)
    # hp.Int('units_1', 8, 16, default=2)
    # hp.Int('units_2', 8, 16, default=2)
    hp.Fixed(name='units_1', value=16)
    hp.Fixed(name='units_2', value=8)
    return hp


def _get_tf_examples_serving_signature(
        model, tf_transform_output: tft.TFTransformOutput):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    # We need to track the layers in the model in order to save it.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()

        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(features.LABEL_KEY)

        raw_features = tf.io.parse_example(serialized_tf_example,
                                           raw_feature_spec)

        # Transform raw features
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    # We need to track the layers in the model in order to save it.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _input_fn(file_pattern, data_accessor,
              tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.

    :param file_pattern:
        List of paths or patterns of input tfrecord files.
    :param data_accessor:
        DataAccessor for converting input to RecordBatch.
    :param tf_transform_output:
        A TFTransformOutput.
    :param batch_size:
        representing the number of consecutive elements of returned
        dataset to combine in a single batch
    :return:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=features.transformed_name(features.LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema).repeat()


# def _build_keras_model(hidden_units, learning_rate):
def _build_keras_model(hparams: kt.HyperParameters):
    """Creates a DNN Keras model for classifying taxi data.

    :param int hidden_units:
        The layer sizes of the DNN (input layer first).
    :param float learning_rate:
        Learning rate of the Adam optimizer.
    :return:
        A keras Model.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)  # noqa: E501
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key,
            num_buckets=features.VOCAB_SIZE + features.OOV_SIZE,
            default_value=0)
        for key in features.transformed_names(features.VOCAB_FEATURE_KEYS)
    ]
    # categorical_columns += [
    #     tf.feature_column.categorical_column_with_identity(
    #         key,
    #         num_buckets=num_buckets,
    #         default_value=0) for key, num_buckets in zip(
    #             features.transformed_names(features.BUCKET_FEATURE_KEYS),
    #             features.BUCKET_FEATURE_BUCKET_COUNT)
    # ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            features.transformed_name(key),
            num_buckets=num_buckets,
            default_value=0) for key, num_buckets in features.BUCKET_FEATURE_DICT.items()  # noqa: E501
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key,
            num_buckets=num_buckets,
            default_value=0) for key, num_buckets in zip(
                features.transformed_names(features.CATEGORICAL_FEATURE_KEYS),
                features.CATEGORICAL_FEATURE_MAX_VALUES)
    ]
    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        # dnn_hidden_units=hidden_units,
        # learning_rate=learning_rate)
        dnn_hidden_units=[hparams.get('units_1'), hparams.get('units_2')],
        # dnn_hidden_units=constants.HIDDEN_UNITS,
        learning_rate=hparams.get('learning_rate'))
    return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units,
                              learning_rate):
    """Build a simple keras wide and deep model.

    :param wide_columns:
        Feature columns wrapped in indicator_column for wide (linear)
        part of the model.
    :param deep_columns:
        Feature columns for deep part of the model.
    :param int dnn_hidden_units:
        The layer sizes of the hidden DNN.
    :param float learning_rate:
        Learning rate of the Adam optimizer.
    :return:
        A Wide and Deep Keras model
    """
    # Keras needs the feature definitions at compile time.

    # Define input layers for numeric keys
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)  # noqa: E501
        for colname in features.transformed_names(
            features.DENSE_FLOAT_FEATURE_KEYS)
    }

    # Define input layers for vocab keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(features.VOCAB_FEATURE_KEYS)
    })

    # Define input layers for bucket keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        # for colname in features.transformed_names(features.BUCKET_FEATURE_KEYS)  # noqa: E501
        for colname in features.transformed_names(features.BUCKET_FEATURE_DICT.keys())  # noqa: E501
    })

    # Define input layers for categorical keys
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in features.transformed_names(
            features.CATEGORICAL_FEATURE_KEYS)
    })

    # Concatenate numeric inputs
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)

    # Create deep dense network for numeric inputs
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)

    # Concatenate categorical inputs
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    # Create shallow dense network for categorical inputs
    wide = tf.keras.layers.Dense(128, activation='relu')(wide)

    # Define output
    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(
            tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    # Create model
    model = tf.keras.Model(input_layers, output)

    # Define training parameters
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Print model summary
    model.summary(print_fn=logging.info)

    return model


# TFX Tuner will call this function
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.

    :param fn_args:
        Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    :return:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting
                    the model , e.g., the training and validation dataset.
                    Required args depend on the above tuner's implementation.
    """
    # Define tuner search strategy
    tuner = kt.Hyperband(_build_keras_model,
                         hyperparameters=_get_hyperparameters(),
                         objective='val_binary_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='kt_hyperband')

    # Load transform output
    tf_transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_graph, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_graph, constants.EVAL_BATCH_SIZE)

    return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'callbacks': [stop_early],
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


# TFX Trainer will call this function
def run_fn(fn_args: tfx.components.FnArgs) -> None:
    """Train the model based on given args.

    :param fn_args:
        Holds args used to train the model as name/value pairs.
        Refer here for the complete attributes:
        https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py
    """

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create batches of data
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, constants.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_output, constants.EVAL_BATCH_SIZE)

    # Load best hyperparameters
    # hparams = fn_args.hyperparameters.get('values')
    if fn_args.hyperparameters:
        hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is
        # removed from the pipeline. User can also inline the hyperparameters
        # directly in _build_keras_model.
        hparams = _get_hyperparameters()
    logging.info('HyperParameters for training: %s' % hparams.get_config())

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(hparams)

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model = _build_keras_model(hidden_units=constants.HIDDEN_UNITS,
    #                                learning_rate=constants.LEARNING_RATE)

    # Callback for TensorBoard, write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(train_dataset,
              steps_per_epoch=fn_args.train_steps,
              validation_data=eval_dataset,
              validation_steps=fn_args.eval_steps,
              callbacks=[tensorboard_callback],
              epochs=constants.NUM_EPOCHS)

    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf',
               signatures=signatures)
