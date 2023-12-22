import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose,
    Input, LeakyReLU, ReLU, UpSampling2D, MaxPooling2D, concatenate)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from skimage.color import rgb2lab, lab2rgb


# Parameters to change
IMAGE_SIZE = 32
EPOCHS = 1000
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100


def generate_dataset(images):
    X = []
    Y = []
    for i in images:
        lab_image_array = rgb2lab(i / 255)
        x = lab_image_array[:, :, 0]
        y = lab_image_array[:, :, 1:]
        y /= 128  # Normalize
        X.append(x.reshape(IMAGE_SIZE, IMAGE_SIZE, 1))
        Y.append(y)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

def load_data():
    (train_images, _), (test_images, _) = cifar10.load_data()
    
    # Filter out grayscale images
    train_images = [img for img in train_images if not is_grayscale(img)]
    test_images = [img for img in test_images if not is_grayscale(img)]

    X_train, Y_train = generate_dataset(train_images)
    X_test, Y_test = generate_dataset(test_images)
    
    return X_train, Y_train, X_test, Y_test

def is_grayscale(img):
    return np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,0], img[:,:,2])

def make_unet_generator_model():
    def conv_block(x, filters, kernel_size, strides=1):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def deconv_block(x, filters, kernel_size):
        x = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Encoder
    e1 = conv_block(inputs, 32, 4, strides=2)
    e2 = conv_block(e1, 64, 4, strides=2)
    e3 = conv_block(e2, 128, 4, strides=2)
    e4 = conv_block(e3, 256, 4, strides=2)

    # Decoder
    d1 = deconv_block(e4, 128, 4)
    d1 = concatenate([d1, e3])
    d2 = deconv_block(d1, 64, 4)
    d2 = concatenate([d2, e2])
    d3 = deconv_block(d2, 32, 4)
    d3 = concatenate([d3, e1])

    output_layer = Conv2DTranspose(2, 4, strides=2, padding='same', activation='tanh')(d3)

    return Model(inputs=inputs, outputs=output_layer)

def make_patchgan_discriminator_model():
    def conv_block(x, filters, kernel_size, strides=1):
        # x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    d1 = conv_block(inputs, 64, 4, strides=2)
    d2 = conv_block(d1, 128, 4, strides=2)
    d3 = conv_block(d2, 256, 4, strides=2)
    d4 = conv_block(d3, 512, 4, strides=1) # stride 1 for PatchGAN

    output_layer = Conv2D(1, 4, strides=1, padding='same')(d4)

    model = Model(inputs=inputs, outputs=output_layer)
    
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='patchgan_model.png')

    return model

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output) * 0.9, disc_real_output)  #Label Smoothing
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    # Mean Absolute Error
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def visualize_results(epoch, num_samples=5):
    idx = np.random.randint(0, X_test.shape[0], num_samples)
    real_gray_images = X_test[idx]
    real_color_images = Y_test[idx] * 128

    generated_color_images = generator.predict(real_gray_images) * 128

    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        # Show Grayscale Images
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(real_gray_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

        # Show Real Color Images
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(lab2rgb(np.concatenate([real_gray_images[i], real_color_images[i]], axis=-1)))
        plt.axis('off')

        # Show Generated Color Images
        plt.subplot(3, num_samples, i + 1 + 2*num_samples)
        plt.imshow(lab2rgb(np.concatenate([real_gray_images[i], generated_color_images[i]], axis=-1)))
        plt.axis('off')

    plt.tight_layout()

    output_dir = "output_images_cifar"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")

    plt.close()

def save_training_plots(epoch, gen_losses, disc_losses, disc_accuracies):
    if not os.path.exists('training_plots_cifar'):
        os.makedirs('training_plots_cifar')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Show losses
    ax1.plot(gen_losses, label='Generator Loss')
    ax1.plot(disc_losses, label='Discriminator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Show discriminator accuracy
    ax2.plot(disc_accuracies, label='Discriminator Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.suptitle(f'Epoch: {epoch}')
    fig.savefig(f'training_plots_cifar/epoch_{epoch}.png')
    plt.close(fig)

def discriminator_accuracy(disc_real_output, disc_generated_output):
    real_probs = tf.sigmoid(disc_real_output)
    generated_probs = tf.sigmoid(disc_generated_output)

    real_accuracy = tf.reduce_mean(tf.cast(tf.math.round(real_probs), tf.float32))
    generated_accuracy = tf.reduce_mean(tf.cast(tf.math.round(1.0 - generated_probs), tf.float32))
    total_accuracy = (real_accuracy + generated_accuracy) / 2.0
    
    return total_accuracy

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(tf.keras.layers.concatenate([input_image, target]), training=True)
        disc_generated_output = discriminator(tf.keras.layers.concatenate([input_image, gen_output]), training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    disc_accuracy = discriminator_accuracy(disc_real_output, disc_generated_output)

    return gen_total_loss, disc_loss, disc_accuracy

def train_gan(epochs):
    gen_losses = []
    disc_losses = []
    disc_accuracies = []

    for epoch in range(epochs):
        gen_loss_total = disc_loss_total = disc_acc_total = count = 0
        for input_image, target in train_dataset:
            gen_loss, disc_loss, disc_acc = train_step(input_image, target)
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            disc_acc_total += disc_acc
            count += 1

            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            disc_accuracies.append(disc_acc)

        print(f"Epoch: {epoch}, D-Loss: {disc_loss_total/count}, G-Loss: {gen_loss_total/count}, D-Acc: {disc_acc_total/count}")
        if epoch % 5 == 0:
            visualize_results(epoch)
            save_training_plots(epoch, gen_losses, disc_losses, disc_accuracies)

            output_dir = "models_cifar"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            generator.save('models_cifar/generator_model.h5')
            discriminator.save('models_cifar/discriminator_model.h5')

X_train, Y_train, X_test, Y_test = load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE)

LAMBDA = 100
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_unet_generator_model()
discriminator = make_patchgan_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4/8, beta_1=0.5)

train_gan(epochs=EPOCHS)