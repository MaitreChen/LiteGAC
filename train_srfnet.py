from datetime import datetime
from tqdm import tqdm
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

from models.srfnet import SRFNet, SRFBasicBlock
from utils.cnn_dataset import get_dataloader
from utils.setup import *


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="0", help='GPU ids to use.')

    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--num-epoch', type=int, default=80, help='number of epochs for training (default: 100)')
    parser.add_argument('--optim-policy', type=str, default='adam', help='optimizer for training. [sgd | adam]')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 3e-4')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight-decay for training. [le-4 | 1e-6]')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='whether to use half precision training')
    parser.add_argument('--seed', type=int, default=47, help='random seed. [47 | 3407 | 1234]')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')

    parser.add_argument('--exp-name', type=str, default='',
                        help='exp name for training')

    return parser


def he_initialize_model(model):
    print('==> Initializing model..')
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return correct / total


def train(opt,
          model,
          train_dataloader,
          val_dataloader,
          device):
    print('==> Training started..')

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(1, opt.num_epoch + 1):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), postfix=dict, mininterval=0.3)
        for i, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch}/{opt.num_epoch}]")
            loop.set_postfix({'total_loss': total_loss / (i + opt.batch_size)})
            loop.update(2)

        scheduler.step()

        train_acc = evaluate(model, train_dataloader, device)
        val_acc = evaluate(model, val_dataloader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_save_path, opt.model_name + '.pth').replace('\\', '/'))
            print(
                f"Save best model to checkpoints file!\t Train Acc: {train_acc * 100.:.2f}% Val Acc: {val_acc * 100.:.2f}%")

    print('==> Training ended...')


if __name__ == '__main__':
    opt = get_argparse().parse_args()
    print(opt)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    set_random_seed(opt.seed)

    ckpt_save_path = os.path.join('checkpoints/SRFNet', opt.exp_name).replace('\\', '/')
    os.makedirs(ckpt_save_path, exist_ok=True)
    print(f"Checkpoints results to {ckpt_save_path}.")

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_save_path = os.path.join(f'logs/SRFNet', f'{opt.exp_name}-{current_time}').replace('\\', '/')
    os.makedirs(log_save_path, exist_ok=True)

    setup_logging(log_filename=os.path.join(log_save_path).replace('\\', '/'))
    log_hyperparameters(args=opt)

    dataset_c_root = './DatasetC'

    train_dataloader, val_dataloader = get_dataloader(opt.batch_size)

    model = SRFNet(SRFBasicBlock, [1, 1, 3, 1]).cuda()

    train(opt,
          model,
          train_dataloader,
          val_dataloader,
          DEVICE,
          )
