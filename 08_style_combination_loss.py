import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from keras.applications import vgg19
from tensorflow import keras

OUT_FOLDER = "output/style_tests/"

style_layer_names = [
    ("block1_conv1", 0.4),
    ("block2_conv1", 0.3),
    ("block3_conv1", 0.2),
    ("block4_conv1", 0.05),
    ("block5_conv1", 0.05),
]
content_layer_name = "block5_conv2"


class StyleTransfer:
    iterations = 4000
    style_update_freq = 100
    save_freq = 100
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8
    sampling_mode = "SEQ"  # RAND

    def __init__(self, prefix,
                 base_img_path,
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

        self.base_img_path = base_img_path

        self.width, self.height = keras.preprocessing.image.load_img(self.base_img_path).size
        self.img_nrows = 400
        self.img_ncols = int(self.width * self.img_nrows / self.height)

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

    def preprocess_image(self, image_path):
        # Util function to open, resize and format pictures into appropriate tensors
        img = keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_nrows, self.img_ncols)
        )
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(self, x):
        # Util function to convert a tensor into a valid image
        x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def gram_matrix(self, x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(self, style, combination):
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(self, base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(self, x):
        a = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, 1:, : self.img_ncols - 1, :]
        )
        b = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, : self.img_nrows - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))

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
        loss = loss + self.content_weight * self.content_loss(
            base_image_features, combination_features
        )
        for layer_name, weight in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_reference_features, combination_features)
            loss += weight * sl # (self.style_weight / len(style_layer_names))

        loss += self.total_variation_weight * self.total_variation_loss(combination_image)
        return loss

    def compute_loss_combination(self, combination_image, base_image, style_reference_images):
        combined_loss = 0.0
        for style_ref_img in style_reference_images:
            combined_loss += self.compute_loss(combination_image, base_image, style_ref_img)
        return combined_loss / len(style_reference_images)

    @tf.function
    def compute_loss_and_grads(self, combination_image, base_image, style_reference_images):
        with tf.GradientTape() as tape:
            loss = self.compute_loss_combination(combination_image, base_image, style_reference_images)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def _get_all_style_ref(self):
        for f in self.styles:
            yield self.preprocess_image(f)

    def _save_run_params(self):
        run_params = {}
        for name, val in self.__dict__.items():
            if type(val) in [str, int, float]:
                run_params[name] = val
        with open(self.out_folder + "params.json", "w") as f:
            json.dump(run_params, f, indent=4)

    def train(self):
        st = time.time()
        base_image = self.preprocess_image(self.base_img_path)
        style_reference_images = list(self._get_all_style_ref())
        combination_image = tf.Variable(self.preprocess_image(self.base_img_path))

        for i in range(1, self.iterations + 1):
            loss, grads = self.compute_loss_and_grads(
                combination_image, base_image, style_reference_images
            )
            self.optimizer.apply_gradients([(grads, combination_image)])
            if i % self.save_freq == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = self.deprocess_image(combination_image.numpy())
                fname = self.out_folder + "at_iteration_%d.png" % i
                keras.preprocessing.image.save_img(fname, img)
        self._save_run_params()
        print(f"{self.prefix} training ended in {time.time() - st} sec")


if __name__ == "__main__":
    # StyleTransfer(prefix="test", style_update_freq=100, save_freq=100,
    #               iterations=1000, total_variation_weight=1e-6, style_weight=1e-6,
    #               content_weight=2.5e-8, styles_path="input/styles/test2/",
    #               initial_learning_rate=90.0, decay_steps=100, decay_rate=0.96).train()
    jobs = [
        StyleTransfer(prefix="test_107_marq", base_img_path="input/myplace2.jpg",
                      style_update_freq=100, save_freq=100,
                      iterations=1000, total_variation_weight=1e-6, style_weight=1e-7,
                      content_weight=2.5e-8, styles_path="input/styles/test7/",
                      initial_learning_rate=90.0, decay_steps=100, decay_rate=0.99,
                      sampling_mode="RAND")
    ]
    for st in jobs:
        try:
            st.train()
        except Exception as e:
            print(e)
