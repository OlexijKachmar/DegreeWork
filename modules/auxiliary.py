import random
import os
import torch
import numpy as np
import torchvision
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

hyper_params = {
    'N_CLASSES': 6,
    'RANDOM_SEED': 42,
    'LEARNING_RATE': 0.001,
    'BATCH_SIZE': 32,
    'N_EPOCHS': 15,
    'WEIGHT_DECAY': 1e-6
}


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_loss_precision(base_acc_loss):
    fig = make_subplots(rows=2, cols=1, start_cell="bottom-left", subplot_titles=('Loss by epoch', 'Accuracy by epoch'),
                        row_heights=[500, 500])
    train_acc, test_acc, train_l, test_l = base_acc_loss[:, 0], base_acc_loss[:, 1], base_acc_loss[:, 2], base_acc_loss[
                                                                                                          :, 3]

    fig.add_trace(go.Scatter(x=np.arange(len(train_acc)) + 1, y=train_acc,
                             mode='lines+markers',
                             name=f'train accuracy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(test_acc)) + 1, y=test_acc,
                             mode='lines+markers',
                             name=f'test accuracy'), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(train_l)) + 1, y=train_l,
                             mode='lines+markers',
                             name=f'train loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(test_l)) + 1, y=test_l,
                             mode='lines+markers',
                             name=f'test loss'), row=1, col=1)

    fig.update_xaxes(title_text="Epoch number", row=1, col=1)

    fig.update_yaxes(title_text="Loss value", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy score", row=2, col=1)
    fig.update_layout(height=800, width=1000)
    fig.show()

def show_transformed_images(dataset):
    #     torch.manual_seed(1)
    torch.manual_seed(3)
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
