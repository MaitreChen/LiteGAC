from datetime import datetime
from time import time
import argparse
import yaml
import os

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch.optim
from torch import nn

from models.stablegan import Generator, Discriminator
from utils.gan_dataset import get_gan_dataloader
from utils.plot_utils import plot_g_d_loss, plt
from utils.setup import *


def get_argparse():
    """
    Initialize and configure command - line arguments for the training script.

    Returns:
        argparse.ArgumentParser: Configured argument parser object.
    """
    parser = argparse.ArgumentParser()

    # Dataset selection
    parser.add_argument('--dataset_name', type=str, default='',
                        help='Path to the dataset directory', choices=['datasetA', 'datasetB'])

    # Model and training parameters
    parser.add_argument('--image-size', type=int, default=128, help='image size (default: 64)')
    parser.add_argument('--z-dim', type=int, default=100, help='z-dim of generator (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--num_epochs', type=int, default=800, help='number of epochs for training (default: 200)')
    parser.add_argument('--g-lr', type=float, default=0.0001, help='learning rate of generator (default: 0.0002)')
    parser.add_argument('--d-lr', type=float, default=0.0004, help='learning rate of discriminator (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta of generator (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta of discriminator (default: 0.999)')

    # Weight clipping and random seed
    parser.add_argument('--weight-clip', type=float, default=0.01,
                        help='clipping value for discriminator weights (default: 0.01)')
    parser.add_argument('--seed', type=int, default=47, help='random seed 1234 | 47 | 3407')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')

    # Checkpoint and experiment configuration
    parser.add_argument('--save-interval', type=int, default=100, help='interval of saving checkpoints')
    parser.add_argument('--exp-name', type=str, default='',
                        help='exp name for checkpoints directory')

    return parser


def weights_init(m):
    """
    Initialize the weights of linear layers with a normal distribution.

    Args:
        m (nn.Module): PyTorch module to initialize weights for.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def train(opt, G, D, dataloader):
    """
    Train a GAN (Generative Adversarial Network) using the provided generator and discriminator.

    Args:
        opt (argparse.Namespace): Command - line arguments.
        G (nn.Module): Generator model.
        D (nn.Module): Discriminator model.
        dataloader (DataLoader): Data loader for the training dataset.

    Returns:
        tuple: Lists of generator and discriminator losses over training steps.
    """
    print('==> Training started..')

    # Move models to the specified device (GPU or CPU)
    DEVICE = 'cuda' if opt.use_gpu and torch.cuda.is_available() else 'cpu'
    G.to(DEVICE)
    D.to(DEVICE)

    # Set models to training mode
    G.train()
    D.train()

    batch_size = dataloader.batch_size
    num_epochs = opt.num_epochs
    WEIGHT_CLIP = opt.weight_clip

    # Initialize optimizers for generator and discriminator
    g_optimizer = Adam(G.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2))
    d_optimizer = Adam(D.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2))

    # Initialize learning rate schedulers
    g_scheduler = CosineAnnealingLR(g_optimizer, T_max=num_epochs)
    d_scheduler = CosineAnnealingLR(d_optimizer, T_max=num_epochs)

    G_losses = []
    D_losses = []

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, images in enumerate(dataloader):
            # Move images to the specified device
            images = images.to(DEVICE)
            mini_batch_size = images.size()[0]

            # Train discriminator with real images
            d_out_real = D(images)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # Generate fake images
            input_z = torch.randn(mini_batch_size, opt.z_dim).to(DEVICE)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)

            # Train discriminator with fake images
            d_out_fake = D(fake_images)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            epoch_d_loss += d_loss.item()

            # Update discriminator parameters
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # Clip discriminator weights
            for p in D.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train generator
            d_out_fake = D(fake_images)
            g_loss = -d_out_fake.mean()

            # Update generator parameters
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            D_losses.append(d_loss.item())
            G_losses.append(g_loss.item())

        t_epoch_finish = time()

        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()

        print(
            f'epoch {epoch} || Epoch_D_Loss:{epoch_d_loss / batch_size:.4f} || Epoch_G_Loss:{epoch_g_loss / batch_size:.4f}')
        print(f'Time taken: {t_epoch_finish - t_epoch_start:.3f} sec.')

    print('==> Training ended..')

    return G_losses, D_losses


if __name__ == "__main__":
    # Parse command - line arguments
    opt = get_argparse().parse_args()
    print(opt)

    # Determine the device to use (GPU or CPU)
    DEVICE = 'cuda' if opt.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')

    # Set random seed for reproducibility
    set_random_seed(opt.seed)

    # Create directory to save generated images
    IMAGE_SAVE_DIR = f'outs/stablegan/{opt.exp_name}'
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

    # Load dataset configuration
    if opt.dataset_name == 'datasetA':
        CONFIG_PATH = 'configs/config1.yaml'
    elif opt.dataset_name == 'datasetB':
        CONFIG_PATH = 'configs/config2.yaml'
    else:
        raise ValueError(f"Invalid dataset name: {opt.dataset_name}")
    # Save loss values to a text file
    # Plot generator and discriminator losses
    # Train the GAN
    # Initialize model weights
    # Initialize generator and discriminator models
    # Get training data loader
    # Set up logging
    # Create directory to save logs
    # Create directory to save checkpoints
    # Create hyperparameter table

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
    DATA_DIR = yaml_info['root1']
    CLASS_NAME = yaml_info['class_name']

    create_hyperparameter_table(opt)

    ckpt_save_dir = os.path.join(f'./checkpoints/stablegan', opt.exp_name).replace('\\', '/')
    os.makedirs(ckpt_save_dir, exist_ok=True)
    print(f"Checkpoints results to {ckpt_save_dir}.")

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_save_path = os.path.join(f'logs/stablegan/', f'{opt.exp_name}-{current_time}').replace('\\', '/')
    os.makedirs(log_save_path, exist_ok=True)

    setup_logging(log_filename=os.path.join(log_save_path).replace('\\', '/'))
    log_hyperparameters(opt)

    train_dataloader = get_gan_dataloader(opt.dataset_name,
                                          DATA_DIR,
                                          CLASS_NAME,
                                          opt.batch_size,
                                          opt.num_workers)

    G = Generator(z_dim=opt.z_dim, image_size=opt.image_size)
    D = Discriminator(image_size=opt.image_size)

    G.apply(weights_init)
    D.apply(weights_init)

    G_loss_set, D_loss_set = train(opt,
                                   G, D,
                                   train_dataloader,
                                   IMAGE_SAVE_DIR)

    plot_g_d_loss(G_loss_set, D_loss_set)

    loss_path = os.path.join(log_save_path, 'loss_log.txt')
    with open(loss_path, 'w') as f:
        f.write("g_loss\td_loss\n")
        for g, d in zip(G_loss_set, D_loss_set):
            f.write(f"{g:.6f}\t{d:.6f}\n")
    print(f"Loss values saved to {loss_path}")
