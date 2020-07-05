import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def load_data_files():
    """In this function we use the Pandas read_csv method to 
    convert the csv files in dataframes.
    
    We then split the dataframes into training and testing sets 
    
    Next we combine the X_train and y_train into one dataframe 
    
    in the end we return the X_train, X_test, y_train, y_test and merged dataframe"""
    
    #read csvs
    import pandas as pd
    features = pd.read_csv('../../data/training_set_values.csv')
    labels = pd.read_csv('../../data/training_set_labels.csv')
    
    #train and test split, random_state of 2020, test_size = 25%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=2020, test_size=0.25)
    
    #combine X_train and y_train on common ids
    df = pd.merge(y_train, X_train) 
    
    return X_train, X_test, y_train, y_test, df


def check_for_nans(df):
    """This function takes in a dataframe and calculates the 
    nan counts in all the columns and returns a new dataframe with 
    the names of columns containing nans as index and their nan counts
    """
    
    nan_counts = pd.DataFrame(df.isna().sum(), columns=['nan_count'])
    return nan_counts[nan_counts['nan_count'] > 0]


def fillnan_as_unknown(df, name):
    """This function takes a dataframe and one column name and
    convert the Nans in that column to 'unknown'
    """
    
    df[name].fillna('unknown', inplace=True)


def fill_all_nans(df):
    """This function takes in a dataframe, checks it for NaNs
    and converts those missing values as unknowns """
    
    column_names = list(check_for_nans(df).index)
    for name in column_names:
        fillnan_as_unknown(df, name)
        

def create_target(df):
    le = LabelEncoder()
    target = le.fit_transform(df['status_group'])
    target = pd.DataFrame(target, columns=['target'])
    df = pd.concat([target, df], axis=1)
    df.drop('status_group', axis=1, inplace=True)
    classes_dict = {k:v for k,v in zip(
                            le.transform(['functional', 'functional needs repair', 'non functional']), 
                            ['functional', 'functional needs repair', 'non functional'])}
    
    return classes_dict, df

def load_data_and_split():
    """In this function we use the Pandas read_csv method to 
    convert the csv files in dataframes.
    
    We then split the dataframes into training and testing sets 
    
    Next we combine the X_train and y_train into one dataframe 
    
    in the end we return the X_train, X_test, y_train, y_test and merged dataframe"""
    
    #read csvs
    import pandas as pd
    features = pd.read_csv('../../data/training_set_values.csv')
    labels = pd.read_csv('../../data/training_set_labels.csv')
    
    #train and test split, random_state of 2020, test_size = 25%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=2020, test_size=0.25)
    
    #combine X_train and y_train on common ids
    df = pd.merge(y_train, X_train) 
    
    return X_train, X_test, y_train, y_test, df


{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}