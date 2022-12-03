paths = {

    'sample_data' : './dataset/train_gwm_100k.parquet', #First 100K rows of the total train data

    # Data with groupwise mean filling of nan values
    'total_data' : './dataset/train_gwm_mean_total.parquet',
    'train_data' : './dataset/train_gwm_train.parquet',
    'test_data' : './dataset/train_gwm_test.parquet',
    #'val_data' : '',
    'labels_data' : './dataset/train_labels.csv',
    
    # Data with groupwise mean filling of nan values -->> Aggregated based on mean of values
    'train_data_agg' : './dataset/train_gwm_train_agg_mean.parquet',
    'val_data_agg' : './dataset/train_gwm_test_agg_mean.parquet',
    #'test_data_agg' : '',

    # Data with groupwise mean filling of nan values -->> Aggregated based on mean of values
    # train_gwm_train_agg_median.parquet
    # train_gwm_test_agg_median.parquet
    
    # Data with groupwise mean filling of nan values -->> Aggregated based on mean  +  Aggregated based on mean 
    # train_gwm_train_agg_mega.parquet
    # train_gwm_test_agg_mega.parquet

}

categorical_variables = ["B_30", "B_38", "D_63", "D_64", "D_66", "D_68", "D_114", "D_116", "D_117", "D_120", "D_126"]
binary_columns = ['R_2', 'S_6', 'R_4', 'D_66', 'R_15', 'S_18', 'D_86', 'D_87', 'B_31', 'R_19', 'B_32', 'S_20', 'R_21', 'B_33', 'R_22', 'R_23', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'D_103', 'D_109', 'D_114', 'D_116', 'D_120', 'D_127', 'D_129', 'R_28', 'D_135', 'D_137', 'D_139', 'D_140', 'D_143']
multiclass_columns = ['D_51', 'D_63', 'D_64', 'D_68', 'B_22', 'D_79', 'R_10', 'R_11', 'D_82', 'B_30', 'R_18', 'D_91', 'D_92', 'B_38', 'D_108', 'D_111', 'D_117', 'D_123', 'D_125', 'D_126', 'D_136', 'D_138']






