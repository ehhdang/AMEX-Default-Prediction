import pandas as pd
import numpy as np
import pyarrow
from time import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from scipy import stats
from .constants import *
from src.data_preprocessing import *

def groupwise_mean(df_path , start_row  = None, end_row = None):

    '''
    input: 
        df_path: path to Original parquet data from Raddar,
        start_row: Default = None, Defines start index of the dataframe
        end_row: Default = None, Defines end index of the dataframe
    
    Output: Pandas dataframe with groupwise mean replaced

    Takes roughly 2 hours for a million datapoints, 
    use start_row and end_row to process smaller data points. 
    '''
    #df = pd.read_parquet("../input/amex-data-integer-dtypes-parquet-format/train.parquet")  #shape 5531451 rows × 190 columns
    df = pd.read_parquet(df_path)


    if start_row != None:
        df = df.iloc[start_row:end_row]
    
    columns = df.select_dtypes(exclude='object').columns

    df1 = pd.DataFrame()
    df1['customer_ID'] = df['customer_ID']
    df1['S_2'] = df['S_2']

    start = time()
    df1[columns] = df.groupby("customer_ID")[columns].transform(lambda x: x.fillna(x.mean()))
    print(time() - start)

    return df1
    #df1.to_parquet(f"train_groupwise_mean_{start_row}_{end_row}.parquet")


def get_data(sample = False):
    if sample == True:
        df = pd.read_parquet(paths['sample_data'])  #shape 100k rows × 190 columns 
    else:
        df = pd.read_parquet(paths['total_data'])  #shape 5531451 rows × 190 columns
    return df

def get_train_test_split(method = 'method_2', sample = False):
    '''
    Does 80:20 split based on unique Customer IDs
    '''

    df = get_data(sample = sample)
    
    #df = pd.read_parquet('./train_groupwise_mean_total.parquet')
    customer_id_list = df['customer_ID'].unique().tolist()

    if method == 'method_1':  #Based on data
        df_train_test = df[df['S_2'] > '2018-02-01']
        df_train_train = df[df['S_2'] < '2018-02-01']
    
    if method == 'method_2': # 
        train_customer_id , test_customer_id = train_test_split(customer_id_list , test_size = 0.2 , shuffle = True , random_state=33)

        train_df = df[df['customer_ID'].isin(train_customer_id)]
        #train_df.to_parquet("train_groupwise_mean_train.parquet")
        test_df = df[df['customer_ID'].isin(test_customer_id)]
        #test_df.to_parquet("train_groupwise_mean_test.parquet")

        return train_df , test_df
        

def get_aggregated_data(df , group_by_col='customer_ID' , method = 'mean'):
    if method == 'mean':
        df_aggregated = df.groupby(group_by_col).mean()
    if method == 'median':
        df_aggregated = df.groupby(group_by_col).median()
    
    return df_aggregated


def get_aggregated_train_test_split(sample = False):
    train_df , test_df = get_train_test_split(method = 'method_2', sample = sample)

    train_df_mean = get_aggregated_data(train_df)
    #train_aggregated_mean.to_parquet("train_groupwise_mean_train_agg_mean.parquet")
    test_df_mean = get_aggregated_data(test_df)
    #test_aggregated_mean.to_parquet("train_groupwise_mean_test_agg_mean.parquet")

    #train_mega = pd.merge(train_df_mean ,train_df_median, on = 'customer_ID' , suffixes=('_mean' , '_median'))
    #train_mega.to_parquet("train_groupwise_mean_train_agg_mega.parquet")

    # test_mega = pd.merge(test_df_mean ,test_df_median, on = 'customer_ID' , suffixes=('_mean' , '_median'))
    # test_mega.to_parquet("train_groupwise_mean_test_agg_mega.parquet")

    labels = pd.read_csv(paths['labels_data'])

    train_df_mean = pd.merge(train_df_mean , labels , on = 'customer_ID' , how = 'left')
    test_df_mean = pd.merge(test_df_mean , labels , on = 'customer_ID' , how = 'left')

    return train_df_mean ,test_df_mean


def preprocessing(train_df_mean , test_df_mean):
    '''
    Imputes all the missing values with global mean of that columns
    '''
    x_train = train_df_mean.drop(columns = ['customer_ID' , 'target'])
    y_train = train_df_mean['target']
    imp = SimpleImputer(strategy="mean")
    x_train = pd.DataFrame(imp.fit_transform(x_train) , columns = x_train.columns)

    # Test data
    x_test = test_df_mean.drop(columns = ['customer_ID' , 'target'])
    y_test = test_df_mean['target']
    imp = SimpleImputer(strategy="mean")
    x_test = pd.DataFrame(imp.fit_transform(x_test) , columns = x_test.columns)

    #x_train.shape , x_test.shape
    return x_train , x_test , y_train , y_test


def custom_processing(x_train , x_test , y_train , y_test):
    '''
    Add your own preprocessing here.
    '''
    # Scalind training data
    ss = StandardScaler()
    #mm = MinMaxScaler()
    x_train_processed =  ss.fit_transform(x_train)
    x_test_processed = ss.transform(x_test)
    y_train_processed =  y_train
    y_test_processed = y_test
    return x_train_processed , x_test_processed, y_train_processed , y_test_processed


def get_long_train_labels(labels, train_data):
    """merges labels with training data
    Args:
        labels (pd.DataFrame): (N,D) A dataframe containing the customer_ids, with one entry for each customer
        train_data (pd.DataFrame): (N*12, D) A dataframe containing the customer_ids
    Returns:
        labels_long (pd.DataFrame): (N*12,D) A longer version of the training labels where each entry in labels corresponds to the training data.
    """
    return train_data[['customer_ID']].merge(labels)

###########################################################################################
##################### Section 2: Feature Creation     #####################################
###########################################################################################



# Custom transformation strategies for binary, categorical and continuous variables

def transform_binary_vars(data, transformations = ['last', 'mean']):
    """ returns binary variables aggregated at customer level
    Args:
        data (pd.DataFrame): (N*12, D) A dataframe containing the customer information
    Returns:
        data_agg (pd.DataFrame): (N,B*T) A shorter version of the training data with T transformations applied to each of the B binary variables
    """
    data_b = data[['customer_ID']+ binary_columns].copy()
    data_b = data_b.replace(-1, np.nan) # replace -1's with nans to prevent issues

    data_agg = data_b[['customer_ID']].drop_duplicates()

    if 'mean' in transformations:
        temp = data_b.groupby('customer_ID', as_index = False).aggregate('mean').add_suffix("_binary_mean").rename({"customer_ID_binary_mean":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    if 'last' in transformations:
        temp = data_b.groupby('customer_ID', as_index = False).aggregate('last').add_suffix("_binary_last").rename({"customer_ID_binary_last":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)
    
    # note: mode is extremely slow: a better implementation is needed
    if 'mode' in transformations:
        temp = data_b.groupby('customer_ID', as_index = False).aggregate(stats.mode).add_suffix("_binary_mode").rename({"customer_ID_binary_mode":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    data_agg = data_agg.replace( np.nan,0) # replace nans with zeros to prevent issues
    
    print("transformed binary features")

    return data_agg

def transform_categorical_vars(data, transformations = ['last', 'mean','counts','nunique']):
    """ returns categorical variables aggregated at customer level
    Args:
        data (pd.DataFrame): (N*12, D) A dataframe containing the customer information
    Returns:
        data_agg (pd.DataFrame): (N,C*T) A shorter version of the training data with T transformations applied to each of the C categorical variables
    """
    data_c = data[['customer_ID']+ categorical_variables].copy()
    data_c = data_c.replace(-1,np.nan) # replace -1's with nans to prevent issues
    data_agg = data_c[['customer_ID']].drop_duplicates()

    if 'mean' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('mean').add_suffix("_cat_mean").rename({"customer_ID_cat_mean":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    if 'last' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('last').add_suffix("_cat_last").rename({"customer_ID_cat_last":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    if 'counts' in transformations:
        # replace missing
        data_c_2 = data_c.replace(np.nan,0).copy()
        data_c_2[categorical_variables] = data_c_2[categorical_variables].astype(int)

        # one hot encoding
        enc = OneHotEncoder(handle_unknown='ignore', dtype = int, sparse=False)
        data_c_enc = enc.fit_transform(data_c_2[categorical_variables])
        col_names = enc.get_feature_names_out(categorical_variables)
        temp = pd.DataFrame(data_c_enc)
        temp.columns = col_names
        temp['customer_ID'] = data['customer_ID']

        # aggregate with counts (sum)
        temp = temp.groupby('customer_ID', as_index = False).aggregate('mean').add_suffix("_cat_count").rename({"customer_ID_cat_count":"customer_ID"}, axis = 1)

        # merge with other columns
        data_agg = data_agg.merge(temp)

    data_agg = data_agg.replace( np.nan,0) # replace nans with zeros to prevent issues

    print("transformed categorical features")

    return data_agg


def transform_continuous_vars(data, transformations = ['last', 'mean','max','min','std']):
    """ returns binary variables aggregated at customer level
    Args:
        data (pd.DataFrame): (N*12, D) A dataframe containing the customer information
    Returns:
        data_agg (pd.DataFrame): (N,C*T) A shorter version of the training data with T transformations applied to each of the C continous variables
    """

    # get list of continous variables (these are not in binary or categorical)
    continuous_variables = []
    for col in data.columns[2:]:
        if col not in binary_columns and col not in categorical_variables:
            continuous_variables.append(col)

    data_c = data[['customer_ID']+ continuous_variables].copy()
    data_c = data_c.replace(-1, np.nan) # replace -1's with nans to prevent issues

    data_agg = data_c[['customer_ID']].drop_duplicates()

    # make aggregations
    if 'mean' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('mean').add_suffix("_mean").rename({"customer_ID_mean":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    if 'last' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('last').add_suffix("_last").rename({"customer_ID_last":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)
    if 'min' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('min').add_suffix("_min").rename({"customer_ID_min":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)
    if 'max' in transformations:
        temp = data_c.groupby('customer_ID', as_index = False).aggregate('max').add_suffix("_max").rename({"customer_ID_max":"customer_ID"}, axis = 1)
        data_agg = data_agg.merge(temp)

    # impute missing
    imp = SimpleImputer(strategy="mean")
    x = data_agg.iloc[:,1:]
    data_agg.iloc[:,1:] = pd.DataFrame(imp.fit_transform(x) , columns = x.columns)

    # scale values
    # scaler = StandardScaler()
    # data_agg.iloc[:,1:] = scaler.fit_transform(data_agg.iloc[:,1:])

    print("transformed continuous features")

    return data_agg


def create_features(data, labels):
    # perform transformations using previous functions
    data_binary = transform_binary_vars(data)
    data_cat = transform_categorical_vars(data)
    data_continuous = transform_continuous_vars(data)

    # merge features
    data_agg = data_binary.merge(data_cat)
    data_agg = data_agg.merge(data_continuous)

    # add labels
    data_agg = data_agg.merge(labels)
    X = data_agg.drop(['customer_ID','target'], axis = 1)
    y = np.array(data_agg['target'])

    return X, y


