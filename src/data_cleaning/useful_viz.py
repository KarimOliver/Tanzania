import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score,precision_score,f1_score, confusion_matrix,roc_auc_score
from sklearn.multiclass import BaseEstimator
pd.set_option('display.max_columns', None)
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

def show_tree(model,test_or_train,):

    """This will plot the decision tree for your designated decision tree model need to fix this later so it does not show the extra text"""
    from sklearn.tree import plot_tree
    plt.figure(figsize=(25,10))
    p=plot_tree(model,feature_names=test_or_train.columns,filled=True,rounded=True, fontsize=14)
    return p

def js_drop_unnecessary_feature_columns(df):
    """This function drops all the investigated unnecessary 
    columns from the features dataframe and returns the 
    trimmed datadrame.
    """
    
    df.drop(['id', 'date_recorded', 'recorded_by', 'wpt_name',
             'scheme_name', 'num_private', 'subvillage', 'ward',
              'extraction_type_class', 
             'management_group', 'payment_type', 'quality_group',
             'quantity_group', 'source_type', 'source_class', 
             'waterpoint_type_group','extraction_type_group',"amount_tsh"], 
              axis=1, inplace=True)
    
    return df

def drop_unnecessary_feature_columns(df):
    """This function drops all the investigated unnecessary 
    columns from the features dataframe and returns the 
    trimmed datadrame.
    """
    
    df.drop(['id', 'date_recorded', 'recorded_by', 'wpt_name',
             'scheme_name', 'num_private', 'subvillage', 'ward',
             'amount_tsh', 'extraction_type_class', 'region_code',
             'management_group', 'payment_type', 'quality_group',
             'quantity_group', 'source_type', 'source_class', 
             'waterpoint_type_group', 'installer', 'funder',"extraction_type_group"], 
              axis=1, inplace=True)
    
    return df