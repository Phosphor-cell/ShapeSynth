import torch
import matplotlib
import numpy
from torch.utils.data import DataLoader, Dataset

#CHECKING CUDA DO NOT REMOVE
print(f"CUDA: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU")
#END

