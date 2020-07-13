import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns

def find_corr_for_category(name, df):
    """This function takes in categorical feature name and
    Onehot incodes it and spits out a correlation with the classes
    Reminder: {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}
    So if there's a high positive correlation then that would mean that the 
    feature impacts non-functional waterpoints and if there's high negative correlation
    then the feature impacts functional waterpoints
    """
    
    sub_df = df.loc[:, ['target', name]]
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    single_feature_df = sub_df[[name]]
    ohe.fit(single_feature_df)
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=sub_df.index)
    sub_df = sub_df.drop(name, axis=1)
    sub_df = pd.concat([sub_df, ohe_df], axis=1)
    correlation = sub_df.corr().target.sort_values()
    
    return correlation

def plot_relation_with_target(name, df):
    """ Given a feature name and the df, this function plots out
    a bargraph for the relation between feature and  the target
    with regards to the count
    This works for numbered categorical data
    """
    
    plt.figure(figsize=(20,8))

    counts = df.groupby([name]).target.value_counts(sort=False)

    sns.barplot(x=counts.index, y=counts)
    
    #setting the labels here
    plt.tick_params(axis='x', labelrotation=90, labelsize=15)
    plt.xlabel(name.capitalize(), fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.tight_layout()
    
    return plt.show()

def plot_relation_with_target_normalized(name, df):
    """ Given a feature name and the df, this function plots out
    a bargraph for the relation between feature and  the target
    with regards to the count
    This works for numbered categorical data
    """
    
    plt.figure(figsize=(20,8))

    counts = df.groupby([name]).target.value_counts(sort=False, normalize=True)

    sns.barplot(x=counts.index, y=counts)
    
    #setting the labels here
    plt.tick_params(axis='x', labelrotation=90, labelsize=15)
    plt.xlabel(name.capitalize(), fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.tight_layout()
    
    return plt.show()

def unique_values(features, df):
    """This function takes in a list of features and subsets the
    dataframe and prints out the unique values of the individual
    features
    """
    
    sub_df = df.loc[:, features]
    for feature in features:
        print('Unique values for '+ feature.capitalize() + ' : ', 
              sub_df[feature].unique(),'\n' 
              'Number of unique values = ', 
              len(sub_df[feature].unique()), end='\n\n')


def plot_basin_counts(df):
    """This function plots the count for different classes
    with regards to the basin they belong to. It requires
    the dataframe to have a 'target' column for the classes 
    and a 'basin' column as well
    """
    basin_count = df.groupby('basin').target.value_counts(sort=False)

    plt.figure(figsize=(20,8))
    sns.barplot(x=basin_count.index, y=basin_count, palette=['#1AC7C4', 'teal', '#95190C'])
    plt.tick_params(axis='x', labelrotation=90, labelsize=15)
    plt.title('Counts for different classes of Waterwells across different Basins in Tanzania', fontsize=20)
    plt.xlabel('Basin', fontsize=18)
    plt.ylabel('Target', fontsize=18)
    plt.tight_layout()
    
    return plt.show()











{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
