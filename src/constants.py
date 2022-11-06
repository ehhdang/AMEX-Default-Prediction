paths = {

    'sample_data' : './data/train_gwm_100k.parquet', #First 100K rows of the total train data

    # Data with groupwise mean filling of nan values
    'total_data' : './data/train_gwm_mean_total.parquet',
    'train_data' : './data/train_gwm_train.parquet',
    'test_data' : './data/train_gwm_test.parquet',
    #'val_data' : '',
    'labels_data' : './data/train_labels.csv',
    
    # Data with groupwise mean filling of nan values -->> Aggregated based on mean of values
    'train_data_agg' : './data/train_gwm_train_agg_mean.parquet',
    'val_data_agg' : './data/train_gwm_test_agg_mean.parquet',
    #'test_data_agg' : '',

    # Data with groupwise mean filling of nan values -->> Aggregated based on mean of values
    # train_gwm_train_agg_median.parquet
    # train_gwm_test_agg_median.parquet
    
    # Data with groupwise mean filling of nan values -->> Aggregated based on mean  +  Aggregated based on mean 
    # train_gwm_train_agg_mega.parquet
    # train_gwm_test_agg_mega.parquet

}






