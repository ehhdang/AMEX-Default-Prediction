# AMEX Default Prediction
Group 37: _Hassan Naveed, Aditi Prakash, Emma Dang, Amritpal Singh_

## Introduction/Background
With credit-based spending dominating modern consumer behavior and 70% of Americans using a credit card regularly for payments [5], it is critical for credit issuers to assess the risk of their practice to minimize the impact of credit defaulting and ensure greater sustainability of borrowing and lending in the long term. Manual assessment of financial statements and direct calculation of default probability with models like the Merton model is widely leveraged today by lenders. However, these methods are often too generic to capture credit risk at the correct level of granularity and make oversimplifying assumptions that lead to error during risk assessment. 

## Problem Definition
Decision trees, clustering models, and logistical regression are used today to predict credit defaulting based on data that maps consumer demographic and spending data to the credit risk of lending. These models rely on an unmanageably large variety of features and could continue to improve in accuracy. Our AMEX dataset contains features describing spending, payment, balance, and risk-informative measures for AMEX customers across a sample 18-month period, and the target variable is the probability of the customer's most recent credit card statement being defaulted. The goal of this project is to further explore machine learning techniques for more accurate credit default risk prediction based on a relatively small but effective feature set.

## Data Collection and Preprocessing
### Data Collection
We obtain the data set from the [AMEX Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/overview) competition on Kaggle. According to the official competition site, the data contains profile features for each customer at the statement date. All features have been anonymized, normalized, and fall into these categories:
- D_*: Delinquency features
- S_*: Spend features
- P_*: Payment features
- B_*: Balance features
- R_*: Risk features

The training data contains a total of 190 features and contains 3 types of variables: `float` (185 features), `int` (1 feature), `string` (4 features). There are 5,531,450 data points. An initial observation reveals that __*120/190*__ features contain NaN values. Approximately 75% of the features have less than 10% NaN values. Some features like D_87 and B_39 are heavy in NaN values, with more than 90% of the data points being NaN. An initial reaction would be to discard features like D_87 which has 99.93% NaN values; however, 0.07% of the data points is roughly 4,000 data points. Without any doubt, we need to perform some data preprocessing to predict the values of these NaN values.

The training labels are binary: 1 means default while 0 means otherwise. There are a total of 458,913 labels, each of which corresponds to an unique customer ID. There is no NaN values in the labels.

### Comparing Train and Test distributions

#### Difference in Variance between Train and Test data
(Columns with difference bigger than 0.5 threshold)

|Column  |  train_data  | test_data|
| :---:        |    :----:   |         :---:   |
|B_10   | 4.892 |  11.797|
|D_69   | 23.244  |40.610|
|R_7    | 3.031 |  2.415|
|B_26   | 2.915  | 6.453|
|R_14   | 28.336 | 31.363|

#### Difference in Skewness between Train and Test data
(Columns with difference bigger than 30 threshold)


| Column   | train_data |   test_data |
| :---:        |    :----:   |         :---:   |
| D_49    | 3.514       | 60.036 | 
| B_6         | 93.541     |  48.117 | 
| B_10      |   77.712     |  163.682 |
| S_12       |  165.302      |    16.579 |
| D_69       |  83.585     |  144.859 |
| D_71       |  95.553     |  51.730 |
| B_26       |  57.634     |  100.810 |
| D_83       |  16.959     |  80.116 |
| R_23      |   59.143      | 0.011 |
| S_26       |  24.246     |  79.113 |
| B_40       |  45.920      | 169.337 |


### Data Preprocessing
1. Encode categorical features.

| Feature      | Description | Value Range     |
| :---:        |    :----:   |         :---:   |
| `customer_ID`| customer ID  | 458,913 unique ID  |
| `S_2` | Statement date       | Various datetime values      |
| `D_63` | Deliquency variable       | `['CR', 'CO', 'CL', 'XZ', 'XM', 'XL']`      |
| `D_64` | Deliquency variable       | `['O', 'R', nan, 'U', '-1']`      |

We use an ordinal encoder to encode `D_63` and `D_64` features because deliquency variables tend to follow a logical ordering.

2. Predict missing values.  
For each feature, we replace missing data with the mean of the complete data that has a matching label.

3. Normalize data.
Next, we normalize the data so that it has the range between 0 and 1. 

## Dimensionality Reduction    
Data visualization is an important step in machine learning. With a good visualization, we can discover trends, patterns, insights into the data. In this section, we attempt to visualize the AMEX dataset. This is a challenging task because of the large number of features. To ease this task, we reduce the dimensionality of the data by using _Principle Component Analysis (PCA)_ and _t-distributed stochastic neighbor embedding (t-SNE)_. 

### PCA
PCA identifies the combination of attributes, or principle components in the feature space, that explains the most variance in the data. Here, we plot the cumulative variance explained by the principle components of the AMEX dataset. To capture 95% of the variance, we need at least 100 components. 

![Cumulative Variances](images/pca/cumulative_variance.png)
*Figure 1: Cumulative variances of PCA components.*

The figure below shows the scatter plot of the training dataset projected onto three PCA components that capture the most variance. The data corresponding to the compliance class is mapped to a turquoise color, while the data corresponding to the default class is mapped to a dark orange color. There is a large overlap between the compliance class and the default class, showing the challenge of the classification task.

![3D Data Projection on PCA Components](images/pca/pca_projection_3D.gif)

*Figure 2: Training Data Projection on three PCA Components with the Highest Variance.*

The next figure shows the relationship between the first seven PCA components. The turquoise color represents the compliance-class data, and the dark orange color represents the default-class data. According to the figure, no combination of two features offers a good separation of the two classes. The large amount of overlap suggests that the regression model to separate the two class will be highly nonlinear.

![2D Data Projection on PCA Components](images/pca/pca_projection_2D.png)

*Figure 3: Training Data Projection on seven PCA Components with the Highest Variance.*

### t-SNE

![2D Data Projection on tSNE Components](images/tsne/tsne_projection_2D_amrit.png)

*Figure 4: Training Data Projection on two tSNE Components.*

![3D Data Projection on tSNE Components](images/tsne/tsne_projection_3D_amrit.gif)

*Figure 5: Training Data Projection on three tSNE Components*


## Methods:
### Unsupervised
The role of unsupervised learning will be to understand the hidden data structures for better feature processing. 
-  Clustering algorithms: visualize the data to allow better feature processing.
-  Dimensionality reduction (PCA, tSNE and UMAP): Given a total of 190+ features, methods like tSNE and PCA can help visualize the data points and choose relevant features. Reduced feature count could also help boost training speed for supervised methods.

#### KMEANS
Kmeans algorithm separates data into n clusters that minimizes the distance between the data points and the cluster centroids. Because our problem is a binary classification, we use Kmeans to divide our post-PCA processed data into two clusters and classify each cluster based on the majority of the votes of the k-nearest neighbors. 


### Supervised
This is primarily a Supervised Learning problem that requires binary classification. Currently, due to the large size of the dataset, 10,000 customers were used with 80% used for training and 20% for validation. Gradient Boosted trees and Neural Networks have shown promise in this domain [1,3].
####	Gradient Boosting (GB): 
Boosted trees (available through sklearn) have had a great performance in credit risk modeling. However, since trees cannot make use of temporal information, the features would need to be aggregated at customer level. The current approach used the sum of the feature values across time for each customer.

After training an XGBoost classifier with 100 trees and max_depth of 3, we get the following metrics:

| Metrics      | XG Boost Score     |
| :---:        |    :---:   |
| Precision Score| 0.80 |
| Recall Score |  0.76  |
| F-measure | 0.89   |
| Accuracy Score | 0.78  |
| AUC Score | 0.946 |
| GINI Score (G) | 0.8915 |
| Default rate at 4% (D) | 0.975|
|M | 0.9332|


The metrics G, D and M are defined by the competion and are highlighted in the [next section](https://github.com/ehhdang/AMEX-Default-Prediction#evaluation-metrics)

The confusion matrix with this sample of 2000 customers in the validation set is shown below:

![XGboost Confusion Matrix](images/supervised_learning/confusion_matrix.png)

Below is a visual depiction of the AUC curve, which takes into account the false positive and false negative rates at various thresholds:
![XGboost AUC curve](images/supervised_learning/roc_curve.png)

This good performance on the validation set does prove a simpled un-tuned xgboost to be a difficult baseline to outperform for more advanced models:

#### Neural Networks: 
A similar approach can be followed with Feed-forward networks. The temporal nature of the data makes it suitable for Long Short Term Memory (LSTM) networks, and the fixed number of periodicity might permit the use of transformers.

We not only hope to compare these approaches, but also ensemble them together to get our best performing model.

## Results & Discussion
### Evaluation Metrics
We want to recreate the evaluation metric from the competition: https://www.kaggle.com/competitions/amex-default-prediction/overview/evaluation

For the unsupervised methods, we will use both internal metrics (e.g. Beta-CV, Davies-Bouldin, and Silhouette score) and external metrics (e.g. purity, precision, recall, accuracy) to evaluate our clustering models. 

A good clustering result minimizes the distance between intra-cluster data points while maximizing the distance between inter-cluster data points. Beta-CV is a graph-based metrics that computes the ratio of the mean intra-cluster distance to the mean inter-cluster distance. The smaller the Beta-CV score is, the bettter the cluster result is. Silhouette coefficient measures the relative distance from the closest outer cluster to the average intra-cluster distance. A silhouette coefficient close to 1 implies a good cluster because the intra-cluster points are close to one another but far away from other clusters. A coefficient close to -1 indicates that a sample has been assigned to a wrong cluster as a closer cluster is found. A coefficient around 0 indicates overlapping between clusters. Davies-Bouldin index measures how compact the clusters are compared to the distance between the cluster means. A lower Davies_Bouldin index means a better clustering result.

Because we have access to the ground truths of our training data, we compute some external measures to further evaluate the performance of our clustering models. Purity quantifies the extent to which a cluster contains points from only one ground truth partition. A purity value close to 1 indicates a perfect clustering. In this project, we use maximum matching to avoid matching two clusters to the same partition or class. Purity is also known as precision, which measures the quality of our clusters, such as how precisely each cluster represents the ground truth. Another metrics is recall score, which computes how completely each cluster recovers the ground truths. We also report the F-measure, which is the harmonic mean of precision and recall. F-measure captures both the completeness and the precision of the clustering.

For the supervised methods, we introduce two terms:

- Normalized Gini Coefficient (G). Here is was calculated from the AUC score using the formula
$$GINI = (2*AUC)-1 $$
- Default rate at 4% (D). This captures a Sensitivity/Recall statistic by calculating the portion of defaults in the highest-ranked 4% of predictions.
  
Using **G** and **D** our evaluaton metric **M** is found by:
$$M = 0.5 \cdot(G+D) $$

### Discussion
In the Kaggle competition, the best-performing models achieve scores of 0.80 in this metric, and we hope to achieve accuracy close to that. There appears to be some inconsistency with regards to the training and test data provided by AMEX, as the test data is not merely a random sample of the training data. Instead the test data covers not only a separate set of customers, but also a different time period.

Our initial results show the M score around 0.94 in the validation set, but scores around 0.70 in the competition using the same training. Therefore, validation accuracy is not a true reflection of test accuracy in this setting.

#### KMEANS

Our Kmeans model has a silhouette score of __0.23__, a Beta-CV value of __0.23__, and a Davies-Boulder index of __2.64__. The close-to-zero silhouette score indicates an overlap between the two clusters. The small Beta-CV score suggests that the data points within each cluster are close to one another compared to the distance between the cluster means. The PCA component visualization shows the non-linear separation between the two classes. Being a non-parametric clustering model, KMeans may lack the power to give a finer separation between compliance and default customers.

![Kmeans Confusion Matrix](images/kmeans/confusion_matrix_pca.png)

We do a more in-depth analysis of our Kmeans model by looking at several external metrics. The accuracy of the model is 85% on testing data. Nevertheless, the precision score differs greatly between two clusters. Our KMeans model is more precise in predicting compliance data point than predicting default data points. In the confusion matrix figure above, the default cluster has roughly a similar number of default data points as compliance data points. As a result, the default cluster has a low precision score of 0.69. On the other hand, the compliance cluster has a good precision score of 0.92. The Kmeans model predicts default class poorly, despite giving a good prediction of the compliance class. The table below summarizes other externals metrics on this Kmeans model. 

| External Metrics      | Compliance Cluster | Default Cluster     |
| :---:        |    :----:   |         :---:   |
| Precision Score| 0.92  | 0.69  |
| Recall Score |  0.88      | 0.77      |
| F-measure | 0.89       | 0.85      |
| Accurity Score | 0.85       | 0.85     |


## References
1. [Machine Learning: Challenges, Lessons, and Opportunities in Credit Risk Modelling](https://www.moodysanalytics.com/risk-perspectives-magazine/managing-disruption/spotlight/machine-learning-challenges-lessons-and-opportunities-in-credit-risk-modeling) 
1. [Credit Risk Modeling with Machine Learning](https://towardsdatascience.com/credit-risk-modeling-with-machine-learning-8c8a2657b4c4)
1. [Modelling customers credit card behaviour using bidirectional LSTM neural networks](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00461-7)
1. [Research on Credit Card Default Prediction Based on k-Means SMOTE and BP Neural Network](https://www.hindawi.com/journals/complexity/2021/6618841/)
1. [Percent people with credit cards - Country rankings](https://www.theglobaleconomy.com/rankings/people_with_credit_cards/)

## Proposed Timeline
The project's timeline and task breakdown are detailed in this [Gantt chart](https://docs.google.com/spreadsheets/d/1NwSPawBI_k9x3xHloXmnbROMbCaqwuFalB0XVgNrCJ8/edit?usp=sharing).

## Contribution Table for Project Proposal
 - Hassan Naveed: Methods, Result, and Discussion for the supervised portion.
 - Aditi Prakash: Introduction, Background, and Problem Definition.
 - Emma Dang: GitHub Pages, Proposed Timeline, and Contribution Table.
 - Amritpal Singh: Method, Result, and Discussion for the unsupervised portion.
