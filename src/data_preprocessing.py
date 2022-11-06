import pandas as pd
import pyarrow
from time import time
from sklearn.model_selection import train_test_split

from .constants import paths

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

def get_train_test_split(method = 'method_2'):

    df = get_data(sample = False)
    
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


def get_aggregated_train_test_split():
    train_df , test_df = get_train_test_split(method = 'method_2')

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

def get_long_train_labels(labels, train_data):
    """Create a 2D scatterplot of components and labels

    Args:
        labels (pd.DataFrame): (N,D) A dataframe containing the customer_ids, with one entry for each customer
        train_data (pd.DataFrame): (N*12, D) A dataframe containing the customer_ids
    Returns:
        labels_long (pd.DataFrame): (N*12,D) A longer version of the training labels where each entry in labels corresponds to the training data.
    """
    return train_data[['customer_ID']].merge(labels)
