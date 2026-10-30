
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(model_name="MobileNetV2", image_size=384):
    input_shape = (image_size, image_size, 3)

    if model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        preprocess_input = (
            tf.keras.applications.mobilenet_v2.preprocess_input
        )

    elif model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )

        preprocess_input = (
            tf.keras.applications.resnet50.preprocess_input
        )

    else:
        raise ValueError("Unsupported model name")

    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = layers.Lambda(preprocess_input)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(4, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    return model, base_model

if __name__ == "__main__":
    model, base_model = create_model("MobileNetV2")

    model.summary()

    print("Total parameters:", model.count_params())
    print("Backbone frozen:", not base_model.trainable)
    print("Output shape:", model.output_shape)