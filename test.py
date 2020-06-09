import math
import sys
import pickle
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

from nmt_model import NMT

import torch
import torch.nn as nn
import torch.nn.utils

