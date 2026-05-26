import keras_tuner
import tensorflow as tf
from tensorflow.keras import layers

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

import tensorflow_transform as tft


LABEL_KEY = 'class'

FEATURE_KEYS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

NUM_EPOCHS = 5
BATCH_SIZE = 64


def transformed_name(key):
    return key + "_xf"


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP'
    )


def input_fn(
    file_pattern,
    tf_transform_output,
    num_epochs,
    batch_size
):

    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(hp):

    learning_rate = hp.Choice(
        'learning_rate',
        [1e-4, 5e-4, 1e-3]
    )

    dense_1_units = hp.Int(
        'dense_1_units',
        min_value=32,
        max_value=256,
        step=32
    )

    dense_2_units = hp.Int(
        'dense_2_units',
        min_value=16,
        max_value=128,
        step=16
    )

    dropout_rate = hp.Float(
        'dropout_rate',
        min_value=0.1,
        max_value=0.5,
        step=0.1
    )

    inputs = {}
    encoded_features = []

    for feature_name in FEATURE_KEYS:

        input_layer = tf.keras.Input(
            shape=(1,),
            name=transformed_name(feature_name),
            dtype=tf.float32
        )

        inputs[
            transformed_name(feature_name)
        ] = input_layer

        encoded_features.append(input_layer)

    x = layers.concatenate(encoded_features)

    x = layers.Dense(
        dense_1_units,
        activation='relu'
    )(x)

    x = layers.Dropout(
        dropout_rate
    )(x)

    x = layers.Dense(
        dense_2_units,
        activation='relu'
    )(x)

    x = layers.Dropout(
        dropout_rate
    )(x)

    outputs = layers.Dense(
        1,
        activation='sigmoid'
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


def tuner_fn(fn_args: FnArgs):

    tf_transform_output = tft.TFTransformOutput(
        fn_args.transform_graph_path
    )

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output,
        NUM_EPOCHS,
        BATCH_SIZE
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        NUM_EPOCHS,
        BATCH_SIZE
    )

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,

        objective=keras_tuner.Objective(
            'val_auc',
            direction='max'
        ),

        max_trials=5,
        overwrite=True,
        directory=fn_args.working_dir,
        project_name='mushroom_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'epochs': NUM_EPOCHS
        }
    )