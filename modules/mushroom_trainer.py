import tensorflow as tf
from tensorflow.keras import layers

from tfx.components.trainer.fn_args_utils import FnArgs

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


DEFAULT_HP = {
    'learning_rate': 1e-3,
    'dense_1_units': 64,
    'dense_2_units': 32,
    'dropout_rate': 0.2,
}


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
        label_key=transformed_name(LABEL_KEY),
        num_epochs=num_epochs,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(hp):

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
        hp['dense_1_units'],
        activation='relu'
    )(x)

    x = layers.Dropout(
        hp['dropout_rate']
    )(x)

    x = layers.Dense(
        hp['dense_2_units'],
        activation='relu'
    )(x)

    x = layers.Dropout(
        hp['dropout_rate']
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
            learning_rate=hp['learning_rate']
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


def get_serve_tf_examples_fn(
    model,
    tf_transform_output
):

    model.tft_layer = (
        tf_transform_output.transform_features_layer()
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        ]
    )
    def serve_tf_examples_fn(
        serialized_tf_examples
    ):

        feature_spec = (
            tf_transform_output.raw_feature_spec()
        )

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            feature_spec
        )

        transformed_features = (
            model.tft_layer(parsed_features)
        )

        outputs = model(
            transformed_features
        )

        return {
            'outputs': outputs
        }

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):

    tf_transform_output = (
        tft.TFTransformOutput(
            fn_args.transform_output
        )
    )

    if fn_args.hyperparameters:

        hparams = fn_args.hyperparameters

        hp = {

            'learning_rate': hparams.get(
                'learning_rate',
                DEFAULT_HP['learning_rate']
            ),

            'dense_1_units': int(
                hparams.get(
                    'dense_1_units',
                    DEFAULT_HP['dense_1_units']
                )
            ),

            'dense_2_units': int(
                hparams.get(
                    'dense_2_units',
                    DEFAULT_HP['dense_2_units']
                )
            ),

            'dropout_rate': hparams.get(
                'dropout_rate',
                DEFAULT_HP['dropout_rate']
            ),
        }

        print(
            f"\nMenggunakan hyperparameter "
            f"dari tuner: {hp}\n"
        )
        
    else:

        hp = DEFAULT_HP.copy()

        print(
            f"\nMenggunakan default "
            f"hyperparameter: {hp}\n"
        )

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10,
        batch_size=32
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=10,
        batch_size=32
    )

    model = build_model(hp)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir,
            update_freq='epoch'
        )
    ]

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks
    )

    signatures = {

        'serving_default':
            get_serve_tf_examples_fn(
                model,
                tf_transform_output
            )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )

    print(
        f"\nModel berhasil disimpan "
        f"di: {fn_args.serving_model_dir}\n"
    )