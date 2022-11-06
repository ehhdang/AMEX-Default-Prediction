import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pyarrow
from src.unsupervised import *

train_data = pd.read_parquet('dataset/train.parquet')
train_labels = pd.read_csv('dataset/train_labels.csv')
test_data = pd.read_parquet('dataset/test.parquet')

numerical_data = train_data.iloc[:,2:]
numerical_test_data = test_data.iloc[:,2:]

pca, pca_train_components, pca_test_components = pca_transformer(numerical_data, numerical_test_data, 50)

train_labels_long = train_data[['customer_ID']].merge(train_labels)

visualize_pca_cum_variance(pca)

visualize_pca_2D(pca_train_components, train_labels_long)

visualize_pca_2D(pca_train_components, train_labels_long)