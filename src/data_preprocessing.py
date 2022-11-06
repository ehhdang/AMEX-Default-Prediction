def get_long_train_labels(labels, train_data):
    """Create a 2D scatterplot of components and labels

    Args:
        labels (pd.DataFrame): (N,D) A dataframe containing the customer_ids, with one entry for each customer
        train_data (pd.DataFrame): (N*12, D) A dataframe containing the customer_ids
    Returns:
        labels_long (pd.DataFrame): (N*12,D) A longer version of the training labels where each entry in labels corresponds to the training data.
    """
    return train_data[['customer_ID']].merge(labels)
