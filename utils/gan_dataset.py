from matplotlib import pyplot as plt
from PIL import Image
import os
import yaml

from torchvision import transforms
from torch.utils import data
import torchvision.utils

# Load configuration from YAML file
with open('./configs/config1.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)

# Extract image size, mean, and standard deviation from configuration
IMAGE_SIZE = yaml_info['dcgan_image_size']
MEAN = yaml_info['mean']
STD = yaml_info['std']


def load_image_paths_for_class(data_name, data_dir, class_name):
    """
    Load image file paths for a specific class based on the dataset name.

    Args:
        data_name (str): Name of the dataset, e.g., 'datasetA' or 'datasetB'.
        data_dir (str): Directory path where the dataset is stored.
        class_name (str): Name of the class for which to load images.

    Returns:
        list: List of image file paths.
    """
    img_list = []

    if data_name == 'datasetA':
        # Iterate over training and validation splits
        for data_split in ['train', 'val']:
            split_dir = os.path.join(data_dir, data_split, class_name)

            if not os.path.exists(split_dir):
                continue

            # Iterate over all image files in the split directory
            for img_file in os.listdir(split_dir):
                img_path = os.path.join(split_dir, img_file)
                img_list.append(img_path)

        return img_list
    elif data_name == 'datasetB':
        img_list = []

        train_idx_list_path = os.path.join(data_dir, 'train_covid_idx.txt')

        # Read training index list from file
        with open(train_idx_list_path, 'r') as f:
            train_idx_list = f.read().splitlines()

        # Generate image paths based on index list
        for idx in train_idx_list:
            img_list.append(os.path.join('./data2/processed64', idx))
        return img_list


class ImageTransformGAN():
    """
    Transform class for GAN images. Converts images to tensors and normalizes them.
    """

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ])

    def __call__(self, img):
        """
        Apply the transformation to the input image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        return self.data_transform(img)


class GANDataset(data.Dataset):
    """
    Dataset class for GAN. Loads and transforms images.

    Attributes:
        file_list (list): List of image file paths.
        transform (callable): Transformation to apply to images.
    """

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_gray = img.convert('L')

        img_transformed = self.transform(img_gray)

        return img_transformed


def get_gan_dataloader(data_name, data_dir, class_name, batch_size, num_workers):
    """
      Get a data loader for GAN training.

      Args:
          data_name (str): Name of the dataset.
          data_dir (str): Directory path where the dataset is stored.
          class_name (str): Name of the class for which to load images.
          batch_size (int): Number of samples per batch.
          num_workers (int): Number of worker threads for data loading.

      Returns:
          torch.utils.data.DataLoader: Data loader for the training dataset.
      """
    print('==> Getting dataloader..')

    train_img_list = load_image_paths_for_class(data_name, data_dir, class_name)

    train_dataset = GANDataset(
        file_list=train_img_list, transform=ImageTransformGAN())
    print("dataset: ", len(train_dataset))

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader


def show_images(images):
    """
    Display a grid of images.

    Args:
        images (torch.Tensor): Batch of image tensors.
    """
    grid = torchvision.utils.make_grid(images, padding=5, normalize=True)

    grid = grid.numpy().transpose((1, 2, 0))

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle("Pulmonary Images", fontsize=17)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
