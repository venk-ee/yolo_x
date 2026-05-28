import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

from model import YOLOX
from loss import YOLOXLoss
from data import get_dataloader

# def train_one_epoch():
