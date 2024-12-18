import numpy as np
from PIL import Image
import os
import pickle
from generator import Generator
from discriminator import Discriminator
from adam import AdamOpt
from utils import binary_cross_entropy, binary_cross_entropy_derivative

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10_data():
    data_batches = []
    labels = []
    
    for i in range(1, 6):
        batch = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        data_batches.append(batch[b'data'])
        labels.extend(batch[b'labels'])
    
    data = np.concatenate(data_batches, axis=0)  # (50000 , 3072)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (50000, 32, 32, 3)
    labels = np.array(labels)
    
    # normalize data to [-1, 1]
    data = (data / 127.5) - 1.0
    return data, labels

def save_generated_image(image_data, epoch, output_dir="outputFiles"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # image_data: (N,3,32,32)
    image_data = ((image_data + 1) * 127.5).astype(np.uint8)
    image_data = image_data.transpose(0,2,3,1) # back to (N,32,32,3)

    for i, img_array in enumerate(image_data):
        if i == 0: # save only one image for now
            img = Image.fromarray(img_array)
            img.save(os.path.join(output_dir, f"2epoch_{epoch}_sample_{i}.png"))

def clip_gradients(grads, max_norm=10.0):
    total_norm = 0.0
    for g in grads:
        total_norm += np.sum(g**2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]
    return grads

def get_gen_params(gen):
    return [
        gen.W_fc,      # W_fc
        gen.b_fc,      # b_fc
        gen.deconv1.W, # W1
        gen.deconv1.b, # b1
        gen.deconv2.W, # W2
        gen.deconv2.b, # b2
        gen.deconv3.W, # W3
        gen.deconv3.b  # b3
    ]

def set_gen_params(gen, params):
    gen.W_fc      = params[0]
    gen.b_fc      = params[1]
    gen.deconv1.W = params[2]
    gen.deconv1.b = params[3]
    gen.deconv2.W = params[4]
    gen.deconv2.b = params[5]
    gen.deconv3.W = params[6]
    gen.deconv3.b = params[7]

def get_disc_params(disc):
    return [
        disc.W_fc,    # (4096,1)
        disc.b_fc,    # (1,1)
        disc.conv1.W, # (64,3,4,4)
        disc.conv1.b, # (64,)
        disc.conv2.W, # (128,64,4,4)
        disc.conv2.b, # (128,)
        disc.conv3.W, # (256,128,4,4)
        disc.conv3.b  # (256,)
    ]

def set_disc_params(disc, params):
    disc.W_fc    = params[0]
    disc.b_fc    = params[1]
    disc.conv1.W = params[2]
    disc.conv1.b = params[3]
    disc.conv2.W = params[4]
    disc.conv2.b = params[5]
    disc.conv3.W = params[6]
    disc.conv3.b = params[7]

def train_gan(generator, discriminator, train_data, epochs, batch_size, noise_dim, learning_rate, beta1, beta2):
    gen_optimizer = AdamOpt(lr=learning_rate, beta1=beta1, beta2=beta2)
    disc_optimizer = AdamOpt(lr=learning_rate, beta1=beta1, beta2=beta2)
    
    for epoch in range(epochs):
        # random batch
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_images = train_data[idx]  # (batch_size,32,32,3)
        # transpozycja do (N,C,H,W)
        real_images = real_images.transpose(0,3,1,2)  # (batch_size,3,32,32)

        # -------------DISCRIMINATOR---------------
        # fake image 
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_images = generator.forward(noise)  # (batch_size,3,32,32)

        if epoch % 1 == 0:
            save_generated_image(fake_images, epoch)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # real + fake
        combined_images = np.concatenate([real_images, fake_images], axis=0)
        combined_labels = np.concatenate([real_labels, fake_labels], axis=0)

        # shuffle real + fake
        shuffle_indices = np.random.permutation(combined_images.shape[0])
        combined_images = combined_images[shuffle_indices]
        combined_labels = combined_labels[shuffle_indices]

        # disc forward
        predictions = discriminator.forward(combined_images)

        # disc loss
        disc_loss = binary_cross_entropy(combined_labels, predictions)

        # dL/dout
        grad_output = binary_cross_entropy_derivative(combined_labels, predictions)

        # disc backprop
        dx_D, dgrads_D = discriminator.backward(grad_output)

        # dgrads_D = (dW_fc, db_fc, dW1, db1, dW2, db2, dW3, db3)
        disc_params = get_disc_params(discriminator)

        # update disc
        disc_grads = list(dgrads_D) # tuple to list
        disc_grads = clip_gradients(disc_grads, max_norm=20.0)
        disc_optimizer.step(disc_params, disc_grads)
        set_disc_params(discriminator, disc_params)

        # ----------------GENERATOR-------------------
        if epoch >= -1: # pretrain discriminator if needed
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_images = generator.forward(noise)  # (N,3,32,32)
            predictions = discriminator.forward(fake_images) # (N,1)

            # gen loss
            gen_loss = binary_cross_entropy(real_labels, predictions)

            grad_output = binary_cross_entropy_derivative(real_labels, predictions)
            
            # disc backprop
            dx_GD, dgrads_fake = discriminator.backward(grad_output)

            # gen backprop
            dZ_gen, dgrads_G = generator.backward(dx_GD)

            # dgrads_G = (dW_fc, db_fc, dW1, db1, dW2, db2, dW3, db3)
            gen_params = get_gen_params(generator)
            gen_grads = list(dgrads_G)
            gen_grads = clip_gradients(gen_grads, max_norm=20.0)
            gen_optimizer.step(gen_params, gen_grads)
            set_gen_params(generator, gen_params)

        if epoch % 1 == 0:
            if epoch >= -1:
                gen_grad_norm = np.mean([np.linalg.norm(g) for g in gen_grads])
                print(f"Epoch {epoch}: Generator gradient norm: {gen_grad_norm:.6e}")
            disc_grad_norm = np.mean([np.linalg.norm(g) for g in disc_grads])
            print(f"Epoch {epoch}: Discriminator gradient norm: {disc_grad_norm:.6e}")
            print(f"Epoch {epoch}/{epochs} - Gen loss: {round(gen_loss, 4) if epoch >= -1 else 'N/A'} - Disc loss: {round(disc_loss, 4)}") # if epoch>20 else 'N/A'


data, labels = load_cifar10_data()
print(f"Train data shape: {data.shape}, min: {data.min()}, max: {data.max()}")

noise_dim = 128
batch_size = 32
learning_rate = 0.0002
epochs = 200
beta1 = 0.5
beta2 = 0.999

gen = Generator(noise_dim=noise_dim)
disc = Discriminator()

train_gan(gen, disc, data, epochs, batch_size, noise_dim, learning_rate, beta1, beta2)
