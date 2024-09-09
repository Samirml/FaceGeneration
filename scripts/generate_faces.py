import sys
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_gan
from src.dataset import get_dataloader

#Parameters
num_epochs = 1
batch_size = 128

def generate_faces(dataloader,
                   num_epochs,
                   device,
                   output_file='faces_animation.html'):
    """
    Function to generate faces using GAN and save the result as an animation.
    This is an animation of a sequential learning generator with
    the goal of identifying increasingly new and detailed features from the trained sample.

    Parameters:
    - dataloader: Dataloader for training the GAN.
    - num_epochs: Number of epochs to train the model.
    - device: Device to use for computation (CPU or GPU).
    - output_file: File to save the animation (default is 'faces_animation.html').
    """

    img_list = train_gan(num_epochs, dataloader, device)

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i.numpy(), (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=100, blit=True)
    ani.save(output_file, writer='html', fps=10)

    webbrowser.open(output_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/home/samir/PycharmProjects/FacesGeneration/data'
dataloader = get_dataloader(data_dir, batch_size)
generate_faces(dataloader, num_epochs, device)
