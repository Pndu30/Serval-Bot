import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import math

# Setting up Tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# Setting up WandB
import wandb
wandb.init(project="Serval")

# Hyperparameters
batch_size = 64
block_size = 128
n_embd = 128
head_size = 1
dropout_rate = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'