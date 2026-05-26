import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'class'

CATEGORICAL_FEATURES = [
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-root',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring',
    'veil-type',
    'veil-color',
    'ring-number',
    'ring-type',
    'spore-print-color',
    'population',
    'habitat'
]

def transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    outputs = {}

    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature]
        )

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], 'p'),
        tf.int64
    )

    return outputs
