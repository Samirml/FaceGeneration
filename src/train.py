import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Generator, Discriminator, weights_init
from src.dataset import get_dataloader
import torchvision.utils as vutils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_gan(num_epochs, dataloader, device):
    """
    Function to train a Generative Adversarial Network (GAN) model.

    Parameters:
    - num_epochs: Number of training epochs.
    - dataloader: DataLoader to feed the GAN with real data.
    - device: Device to run the training on (CPU or GPU).

    Returns:
    - img_list: A list of generated images for tracking progress.
    """
    #Parameters
    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    lr = 0.0002
    beta1 = 0.5

    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(ndf, nc).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_batch = data.to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_batch).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Printing results
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t'
                      f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                      f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    # Saving generator and discriminator
    # torch.save(netG.state_dict(), 'generator.pth')
    # torch.save(netD.state_dict(), 'discriminator.pth')
    return img_list
