import os
import random
import pickle
from timeit import default_timer as timer
from tqdm import tqdm, trange
import warnings
warnings.filterwarnings("ignore")


import re
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
nltk.download("stopwords")
from string import punctuation

import pandas as pd
import numpy as np
pd.set_option("display.max_rows",20)
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.utils import class_weight
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# from datasets import load_dataset,Dataset
from transformers import AutoModel, BertTokenizerFast, BertModel
from transformers import XLNetTokenizer, XLNetModel

from torch.utils.data import DataLoader
import torch.nn.init as init



import warnings
from sklearn.exceptions import UndefinedMetricWarning
# Ignore UserWarning and UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)