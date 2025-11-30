"""DCGAN training script for CIFAR-10 truck class.
Run: python gan_cifar10_truck.py --epochs 50 --subset 20000 --latent-dim 128
"""
import argparse
import pathlib
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# --------------------------------------------------
# Generator and Discriminator
# --------------------------------------------------

def build_generator(latent_dim: int = 128) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(4*4*512, use_bias=False)(inputs)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for filters in [256, 128, 64]:
        x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2DTranspose(3, 3, activation='tanh', padding='same')(x)
    return tf.keras.Model(inputs, outputs, name='generator')


def build_discriminator() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32,32,3))
    x = inputs
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, x, name='discriminator')

# --------------------------------------------------
# Data
# --------------------------------------------------

def preprocess(image):
    image = tf.image.resize(image, (32,32))
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def get_truck_dataset(batch_size: int, subset: int | None, seed: int = 42):
    (train_ds, _), info = tfds.load('cifar10', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)
    class_names = info.features['label'].names
    truck_label = class_names.index('truck')
    filtered = train_ds.filter(lambda img, lbl: tf.equal(lbl, truck_label))
    if subset:
        filtered = filtered.take(subset)
    ds = (filtered
          .map(lambda img, lbl: preprocess(img), num_parallel_calls=tf.data.AUTOTUNE)
          .shuffle(5000, seed=seed)
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))
    return ds

# --------------------------------------------------
# Training
# --------------------------------------------------

def train(epochs: int, batch_size: int, latent_dim: int, subset: int | None, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    ds = get_truck_dataset(batch_size, subset)
    fixed_noise = tf.random.normal((16, latent_dim))

    @tf.function
    def step(real_images):
        bs = tf.shape(real_images)[0]
        z = tf.random.normal((bs, latent_dim))
        with tf.GradientTape(persistent=True) as tape:
            fake_images = generator(z, training=True)
            real_logits = discriminator(real_images, training=True)
            fake_logits = discriminator(fake_images, training=True)
            d_loss_real = ce(tf.ones_like(real_logits), real_logits)
            d_loss_fake = ce(tf.zeros_like(fake_logits), fake_logits)
            d_loss = d_loss_real + d_loss_fake
            g_loss = ce(tf.ones_like(fake_logits), fake_logits)
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    for epoch in range(1, epochs+1):
        start = time.time()
        d_losses = []
        g_losses = []
        for batch in ds:
            d_l, g_l = step(batch)
            d_losses.append(d_l)
            g_losses.append(g_l)
        d_mean = tf.reduce_mean(d_losses)
        g_mean = tf.reduce_mean(g_losses)

        # Sample images
        samples = generator(fixed_noise, training=False)
        samples = (samples + 1.0) / 2.0
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4,4, figsize=(4,4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].numpy())
            ax.axis('off')
        fig.suptitle(f'Epoch {epoch}')
        fig.savefig(out_dir / f'samples_epoch_{epoch:03d}.png')
        plt.close(fig)

        print(f'Epoch {epoch}/{epochs} D_loss={d_mean:.4f} G_loss={g_mean:.4f} time={time.time()-start:.1f}s')

    generator.save(out_dir / 'generator_savedmodel')
    discriminator.save(out_dir / 'discriminator_savedmodel')
    print('Training complete. Models saved.')

# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train DCGAN on CIFAR-10 truck class.')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--subset', type=int, default=10000, help='Limit number of truck images (None for all)')
    p.add_argument('--out-dir', type=str, default='artifacts/gan_truck')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    train(args.epochs, args.batch_size, args.latent_dim, args.subset if args.subset > 0 else None, out_dir)

if __name__ == '__main__':
    main()
