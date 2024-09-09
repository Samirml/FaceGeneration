import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

#Parameters
batch_size = 128
image_size = 64

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Class to create a dataloader for the CelebA dataset.

        Parameters:
        - root_dir: Directory where the CelebA images are stored.
        - batch_size: Number of images per batch (default: 128).
        - num_workers: Number of worker processes for loading data (default: 4).

        Returns:
        A DataLoader object to load CelebA dataset images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img_name) for
                            img_name in os.listdir(root_dir) if img_name.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def get_dataloader(root_dir, batch_size=128, num_workers=4):
    dataset = CelebADataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader



