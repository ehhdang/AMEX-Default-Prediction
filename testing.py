import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pyarrow
from src.unsupervised import *
from src.data_preprocessing import *

train_data, test_data = get_aggregated_train_test_split(sample = True)
train_labels = train_data[['customer_ID','target']]

train_data.fillna(0, inplace = True)
test_data.fillna(0, inplace = True)
numerical_data = train_data.iloc[:,2:]
numerical_test_data = test_data.iloc[:,2:]

pca, pca_train_components, pca_test_components = pca_transformer(numerical_data, numerical_test_data, 50)

train_labels_long = train_data[['customer_ID']].merge(train_labels)

visualize_pca_cum_variance(pca)

visualize_pca_2D(pca_train_components, train_labels_long)

visualize_pca_2D(pca_train_components, train_labels_long)