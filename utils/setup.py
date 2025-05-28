import random
import logging

import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()


def setup_logging(log_filename):
    file_name = f'{log_filename}/training_details.txt'

    logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO, format='%(message)s')
    print(f"Logging results to {log_filename}.")


def log_hyperparameters(args):
    logging.info("Hyper parameters:")
    logging.info("===================================")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("===================================")


def log_fid_value(epoch, fid):
    logging.info(f"Epoch {epoch}  Fid value: {fid}")
