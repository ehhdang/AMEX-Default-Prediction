
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# Metrics are available in kaggle post: https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def get_results(test_df , gt , preds , pred_prob):
    output_dict = classification_report(gt , preds, output_dict=True)
    conf_matrix = confusion_matrix(gt, preds)
    output_dict['conf_matrix'] = conf_matrix
    sns.heatmap(conf_matrix, annot=True)
    plt.show()
    
    pred_df = pd.DataFrame()
    pred_df['customer_ID'] = test_df['customer_ID']
    pred_df['prediction'] = pred_prob

    true_df = pd.DataFrame()
    true_df['customer_ID'] = test_df['customer_ID']
    true_df['target'] = gt

    output_dict['amex_metric'] = amex_metric(true_df , pred_df)

    
    return output_dict , pred_df

### TODO: Reconcile this with amex_metric()
# recreates evaluation metric from AMEX: 
def evaluate_model(y_true, y_score, traditional = False):
    # produce traditional metric such as Precision, Recall, F-measure and accuracy
    if traditional:
        threshold = 0.5
        pred_labels  = (y_score[:,1] > 0.5)
        true_positive = np.sum(np.logical_and(y_true, pred_labels))
        false_positive = np.sum(np.logical_and(1-y_true, pred_labels))
        false_negative = np.sum(np.logical_and(y_true, 1-pred_labels))
        true_negative = np.sum(np.logical_and(1-y_true, 1-pred_labels))

        print("Traditional Metrics:")
        print(f"    tp: {true_positive}, fp: {false_positive}, tn: {true_negative}, fn: {false_negative}")
        precision = true_positive / (true_positive + false_positive)
        print("     The precision score is ", precision)
        recall = true_positive / (true_positive + false_negative)
        print("     The recall score is ", recall)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
        print("     The accuracy score is ", accuracy)
        fmeasure = 2/(1/precision + 1/recall)
        print("     F-Measure: ", fmeasure)

    AUC = roc_auc_score(y_true, y_score[:,1])
    print(f"AUC Score: {AUC}")

    # GINI from AUC formula: https://yassineelkhal.medium.com/confusion-matrix-auc-and-roc-curve-and-gini-clearly-explained-221788618eb2
    Gini = (AUC*2)-1
    print(f"GINI Score: {Gini}")

    #get top 4% ratio of positive predictions
    fpr, tpr,_ = roc_curve(y_true, y_score[:,1])
    pred_labels.sort()
    pos_samples = np.sum(y_true)
    neg_samples = len(y_true) - pos_samples
    neg_samples_weighted = 20*neg_samples
    total_samples = pos_samples+neg_samples_weighted
    four_percent = int(0.04*total_samples)
    four_percent_index = np.argmax((fpr * neg_samples_weighted + tpr * pos_samples >= four_percent))

    true_positives_at_four_percent = tpr[four_percent_index]*pos_samples

    D = true_positives_at_four_percent/pos_samples
    print(f"Default rate at 4%: {D}")
    
    M = 0.5*(Gini+D)
    print(f"M: {M}")

    return M


# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_score):
    pred_labels = y_score[:,1]>0.5
    cm = confusion_matrix(y_true, pred_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Compliance", "Default"])
    cm_display.plot()
    plt.show()

# Plot GINI (OR AUC) curve
def plot_GINI(y_true, y_score):

    fpr, tpr,_ = roc_curve(y_true, y_score[:,1])
    pos_samples = np.sum(y_true)
    neg_samples = len(y_true) - pos_samples
    neg_samples_weighted = 20*neg_samples
    total_samples = pos_samples+neg_samples_weighted
    four_percent = int(0.04*total_samples)
    four_percent_index = np.argmax((fpr * neg_samples_weighted + tpr * pos_samples >= four_percent))

    RocCurveDisplay.from_predictions(y_true, y_score[:,1])
    plt.axvline(x = 0.04, color = 'r',linestyle = 'dashed')
    plt.scatter([fpr[four_percent_index]],[ tpr[four_percent_index]], s = 100)
    plt.show()
