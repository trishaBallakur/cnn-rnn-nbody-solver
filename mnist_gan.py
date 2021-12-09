import tensorflow as tf
from tensorflow.keras.layers import *
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display


class MNIST_GAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # create the generator
        self.generator = tf.keras.Sequential()

        # normalizes the data
        self.generator.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU())
        self.generator.add(Reshape((7, 7, 256)))
        assert self.generator.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        # upscales and normalizes
        self.generator.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 7, 7, 128)
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU())

        # upscales and normalizes
        self.generator.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 14, 14, 64)
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU())

        # upscales and returns image
        self.generator.add(
            Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.generator.output_shape == (None, 28, 28, 1)

        # create discriminator
        self.discriminator = tf.keras.Sequential()

        # CNN layer 1
        self.discriminator.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dropout(0.3))

        # CNN layer 2
        self.discriminator.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(LeakyReLU())
        self.discriminator.add(Dropout(0.3))

        # final linear layer of discriminator
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(1))

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train_step(model, images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = model.generator(noise, training=True)

        real_output = model.discriminator(images, training=True)
        fake_output = model.discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

    model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
    model.discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, model.discriminator.trainable_variables))


def train(model, dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(model, image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(model.generator,
                                 epoch + 1,
                                 seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(model.generator,
                             epochs,
                             seed)


if __name__ == '__main__':
    EPOCHS = 50
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    noise_dim = 100
    num_examples_to_generate = 16

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    model = MNIST_GAN()

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    noise = tf.random.normal([1, 100])
    generated_image = model.generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    train(model, train_dataset, EPOCHS)

    anim_file = 'gan_junk/dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
