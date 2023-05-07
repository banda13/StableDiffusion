import os
import random
import time

import numpy as np
import tensorflow as tf
from keras.applications import vgg19
from tensorflow import keras

BASE_IMG_PATH = "input/myplace2.jpg"
OUT_FOLDER = "output/style_tests/"

width, height = keras.preprocessing.image.load_img(BASE_IMG_PATH).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
style_layer_names = [
    "block1_conv1", "block2_conv1",
    "block3_conv1", "block4_conv1", "block5_conv1",
]
content_layer_name = "block5_conv2"


def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


class StyleTransfer:

    iterations = 4000
    style_update_freq = 100
    save_freq = 100
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8
    sampling_mode = "SEQ"  # RAND

    def __init__(self, prefix,
                 iterations,
                 total_variation_weight,
                 style_weight,
                 content_weight,
                 styles_path,
                 style_update_freq,
                 save_freq,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate,
                 sampling_mode):

        print(f"Initializing style transfer: {prefix}")

        self.prefix = prefix
        self.iterations = iterations
        self.total_variation_weight = total_variation_weight
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.style_update_freq = style_update_freq
        self.save_freq = save_freq
        self.sampling_mode = sampling_mode

        self.model = vgg19.VGG19(weights="imagenet", include_top=False)
        self.outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.outputs_dict)

        self.optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps, decay_rate=decay_rate
            )
        )

        self.styles = list(map(lambda l: styles_path + l, os.listdir(styles_path)))
        self.style_img_idx = 0

        self.out_folder = OUT_FOLDER + prefix + "/"
        os.mkdir(self.out_folder)

    def compute_loss(self, combination_image, base_image, style_reference_image):
        input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0
        )
        features = self.feature_extractor(input_tensor)

        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + self.content_weight * content_loss(
            base_image_features, combination_features
        )
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (self.style_weight / len(style_layer_names)) * sl

        loss += self.total_variation_weight * total_variation_loss(combination_image)
        return loss

    @tf.function
    def compute_loss_and_grads(self, combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def _get_next_style_ref(self):
        if self.sampling_mode == "SEQ":
            img = preprocess_image(self.styles[self.style_img_idx])
            if self.style_img_idx < len(self.styles) - 1:
                self.style_img_idx += 1
            else:
                self.style_img_idx = 0
            return img
        else:
            return preprocess_image(random.choice(self.styles))

    def train(self):
        st = time.time()
        base_image = preprocess_image(BASE_IMG_PATH)
        style_reference_image = self._get_next_style_ref()
        combination_image = tf.Variable(preprocess_image(BASE_IMG_PATH))

        for i in range(1, self.iterations + 1):
            loss, grads = self.compute_loss_and_grads(
                combination_image, base_image, style_reference_image
            )
            self.optimizer.apply_gradients([(grads, combination_image)])
            if i % self.save_freq == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = deprocess_image(combination_image.numpy())
                fname = self.out_folder + "at_iteration_%d.png" % i
                keras.preprocessing.image.save_img(fname, img)

            if i % self.style_update_freq == 0:
                style_reference_image = self._get_next_style_ref()
        print(f"{self.prefix} training ended in {time.time() -st} sec")


if __name__ == "__main__":
    # StyleTransfer(prefix="test", style_update_freq=100, save_freq=100,
    #               iterations=1000, total_variation_weight=1e-6, style_weight=1e-6,
    #               content_weight=2.5e-8, styles_path="input/styles/test2/",
    #               initial_learning_rate=90.0, decay_steps=100, decay_rate=0.96).train()
    StyleTransfer(prefix="test_9_Ukiyo", style_update_freq=100, save_freq=100,
                  iterations=1000, total_variation_weight=1e-6, style_weight=1e-7,
                  content_weight=2.5e-8, styles_path="data/wikiart/Ukiyo_e/",
                  initial_learning_rate=80.0, decay_steps=100, decay_rate=0.96,
                  sampling_mode="RAND").train()
