import numpy as np
import cv2
import os
import torch


def resize_grayscale_images_in_directory(directory_path, target_size=(64, 64)):
    """
    Resize all grayscale images in the specified directory to the target size.

    Args:
        directory_path (str): Path to the directory containing images.
        target_size (tuple): Target size (width, height) for resizing images. Default is (64, 64).
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    file_list = os.listdir(directory_path)

    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, target_size)
                cv2.imwrite(file_path, img_resized)

                print(f"Resized '{file_name}' to {target_size}")
            except Exception as e:
                print(f"Error processing '{file_name}': {e}")
        else:
            print(f"Skipping non-image file '{file_name}'")


def decode_to_numpy(x):
    """
    Convert a PyTorch tensor to a NumPy array and normalize it to the range [0, 255].

    Args:
        x (torch.Tensor): Input PyTorch tensor.

    Returns:
        tuple: Original tensor converted to NumPy array and normalized NumPy array.
    """
    x = x.cpu().detach().numpy()
    x_array = (x + 1) / 2
    x_array = (x_array - np.min(x_array)) / (np.max(x_array) - np.min(x_array))
    x_array = x_array * 255.

    return x, x_array


def getStat(train_data):
    """
    Compute the mean and standard deviation of the training data.

    Args:
        train_data (torch.utils.data.Dataset): Training dataset.

    Returns:
        tuple: Lists of mean and standard deviation values for each channel.
    """
    n_channels = 1

    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=False, num_workers=8,
        pin_memory=True)

    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    for X, _ in train_loader:
        for d in range(n_channels):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())
