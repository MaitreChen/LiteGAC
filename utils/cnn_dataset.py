from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    """
    A custom dataset class for loading image data from a text file.
    Each line of the text file should contain an image path and its corresponding label, separated by a space.
    """

    def __init__(self, txt_path, transform=None):
        """
        Initialize the dataset.

        Args:
            txt_path (str): Path to the text file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_paths = []
        self.labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                self.img_paths.append(img_path)
                self.labels.append(int(label))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx].replace("\\", '//')
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.6, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])

    # add your own path
    train_txt_path = ''
    test_txt_path = ''

    train_dataset = CustomDataset(train_txt_path, transform=train_transform)
    test_dataset = CustomDataset(test_txt_path, transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, test_dataloader
