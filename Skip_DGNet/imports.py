import os
import json
import glob
import math
import torch
import joblib

import numpy as np
import torch.nn as nn
import os.path as osp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.checkpoint as cp

from math import exp
from tqdm import tqdm
from skimage.util import crop
from torchvision import models
#from geomloss import SamplesLoss
from psutil import virtual_memory
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
#from kornia.enhance import image_histogram2d
#from scipy.stats import binned_statistic, norm
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
#from skimage.transform import resize, warp, AffineTransform

_VALID_SCENE_TYPES = ('decalage', 'plan')