# AMEX Default Prediction
Group 37: _Hassan Naveed, Aditi Prakash, Emma Dang, Amritpal Singh_

## Introduction/Background
> _TO BE FILLED_

## Problem Definition
> _TO BE FILLED_

## Methods:
### Unsupervised
Role of unsupervised learning will be to understand the hidden data structures for better feature processing. 
1) Clustering algorithms: visualise the data to allow better feature processsing.
2) Dimension reductionality (tSNE and PCA): Given total 200+ features, methods like tSNE and PCA can help visualize the data points and choose relevant features. Reduced feature count could also help boost training speed for supervised methods.


### Supervised
This is primarily a Supervised Learning problem which requires binary classification. The models which have shown promise with previous work [1,3] are:
1)	Gradient Boosting (GB): Boosted trees (available through sklearn) have had great performance in credit risk modelling. However, since trees cannot make use of temporal information, the features would need to be aggregated at customer level.
2)	Neural Networks: A similar approach as (1) can be experimented with Feed-forward networks. The temporal nature of the data make is suitable for Long Short Term Memory (LSTM) networks, and the fixed number of periodicity might permit the use of transformers.

We not only hope to compare these approaches, but also ensemble them together to get our best performing model.

## Results & Discussion
The models would be tested according to the competition metric. This consists of the average of:
-	Normalized Gini Coefficient (G)
-	Default rate at 4% (D). This captures a Sensitivity/Recall statistic by calculating the portion of defaults in the highest ranked 4% of predictions

The best performing models achieve scores of 0.80 in this metric, and we hope to achieve accuracy close to that. In addition, models would also be compared using common binary classification metrics such as AUC, Accuracy, Precision, etc.

## References
1. [Machine Learning: Challenges, Lessons, and Opportunities in Credit Risk Modelling](https://www.moodysanalytics.com/risk-perspectives-magazine/managing-disruption/spotlight/machine-learning-challenges-lessons-and-opportunities-in-credit-risk-modeling) 
1. [Credit Risk Modeling with Machine Learning](https://towardsdatascience.com/credit-risk-modeling-with-machine-learning-8c8a2657b4c4)
1. [Modelling customers credit card behaviour using bidirectional LSTM neural networks](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00461-7)
1. [Research on Credit Card Default Prediction Based on k-Means SMOTE and BP Neural Network](https://www.hindawi.com/journals/complexity/2021/6618841/)

## Proposed Timeline
The project's timeline and task breakdown is detailed in this [Gantt chart](https://docs.google.com/spreadsheets/d/1NwSPawBI_k9x3xHloXmnbROMbCaqwuFalB0XVgNrCJ8/edit?usp=sharing).

## Contribution Table for Project Proposal
 - Hassan Naveed : Methods, Result, and Discussion for the supervised portion.
 - Aditi Prakash : Introduction, Background, and Problem Definition.
 - Emma Dang : GitHub Pages, Proposed Timeline, and Contribution Table.
 - Amritpal Singh : Method, Result, and Discussion for the unsupervised portion.
