import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns


def processed_dataset():
    """This function loads the two datasets from the data folder
    and splits it up into X_train, X_test, y_train, y_test 
    with random_state = 2020, test_size = 25%
    
    Then it trims down X Features and create y labels
    
    Returns: X_train, X_test, y_train, y_test, classes_dict
    """
    #using helper function to split dataset
    X_train, X_test, y_train, y_test = load_data_and_split()
    
    #using helper function to transform missig location specific values
    X_train, X_test = transform_location_values(X_train, X_test)
    
    #using helper function to trim down X features
    X_train = drop_unnecessary_feature_columns(X_train)
    X_test = drop_unnecessary_feature_columns(X_test)
    
    #using helper function to fill NaNs
    fill_all_nans(X_train)
    fill_all_nans(X_test)
    
    #replace 0 in construction_year with its mean
    X_train.construction_year.replace(0, 2000, inplace=True)
    X_test.construction_year.replace(0, 2000, inplace=True)
    
    #using helper function to process y_train and y_test
    y_train, y_test, classes_dict = process_y_sets(y_train, y_test)
    
    return X_train, X_test, y_train, y_test, classes_dict


def load_data_and_split():
    """In this function we use the Pandas read_csv method to 
    convert the csv files into dataframes. These csv files are 
    saved in hte data folder as:
    >>> 'training_set_values.csv' (for X features)
    >>> 'training_set_labels.csv' (for y target)
    
    We then split the dataframes into training and testing sets 
    and return the X_train, X_test, y_train, y_test
    """
    
    #read csvs
    import pandas as pd
    features = pd.read_csv('../../data/training_set_values.csv')
    labels = pd.read_csv('../../data/training_set_labels.csv')
    
    #train and test split, random_state of 2020, test_size = 25%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=2020, test_size=0.25) 
    
    return X_train, X_test, y_train, y_test


def transform_location_values(X_train, X_test):
    """This function goes through the columns latitude, longitude,
    gps_height and population and replaces all the 0s and 1s to nulls
    first. (many longitude values were found to be zero so to 
    impute those we'll convert the latitudes to 0 too)
    
    We then transform these nulls into the means of other 
    similar cases within the same district_code, subvillage, and basin)
    
     """
    trans = ['longitude', 'latitude', 'gps_height', 'population']
    
    #if long is 0, lat becomes 0
    for i in [X_train, X_test]:
        i.loc[i.longitude == 0, 'latitude'] = 0
    
    #replace 0 or 1 with Nan
    for z in trans:
        for i in [X_train, X_test]:
            i[z].replace(0., np.NaN, inplace = True)
            i[z].replace(0, np.NaN, inplace = True)
            i[z].replace(1., np.NaN, inplace = True)
            i[z].replace(1, np.NaN, inplace = True)
        
        #loop through the categories of interest to find the means
        #of interest filling them in order of subvillage, then district
        #then basin
        for j in ['subvillage', 'district_code', 'basin']:
        
            X_train['mean'] = X_train.groupby([j])[z].transform('mean')
            X_train[z] = X_train[z].fillna(X_train['mean'])
            o = X_train.groupby([j])[z].mean()
            fill = pd.merge(X_test, pd.DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
            X_test[z] = X_test[z].fillna(fill)
        
        #if still not found, we'll fill the nulls with the mean of the columns
        X_train[z] = X_train[z].fillna(X_train[z].mean())
        X_test[z] = X_test[z].fillna(X_train[z].mean())
        del X_train['mean']
    
    
    return X_train, X_test


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
             'waterpoint_type_group', 'installer', 'funder', 
             'extraction_type_group'], 
              axis=1, inplace=True)
    
    return df


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


#first drop id
def drop_id_from_y(df):
    """This function drops id column 
    from the y sets
    """
    df.drop('id', axis=1, inplace=True)
    
    return df


#convert `status_group` to classes using LabelEncoder
def create_class_labels_for_y(df1, df2):
    """This function takes the y training set as df1
    and y test set as df2 and creates labels 
    for the target.
    Here we fit on training and transform both the 
    training and the test set
    Returns: training df, test df and label dictionary.
    """
    
    le = LabelEncoder()
    le.fit(df1['status_group'])
    target1 = le.transform(df1['status_group'])
    df1 = pd.DataFrame(target1, columns=['target'])
    
    target2 = le.transform(df2['status_group'])
    df2 = pd.DataFrame(target2, columns=['target'])
    
    
    classes_dict = {k:v for k,v in zip(
                            le.transform(['functional', 'functional needs repair', 'non functional']), 
                            ['functional', 'functional needs repair', 'non functional'])}
    
    return df1, df2, classes_dict



def process_y_sets(df1, df2):
    """This function preprocess the y_train and y_test
    by dropping the id column and converting the status_group
    to classes 
    
    Returns: y_train, y_test and label dictionary.
    """
    
    #helper functio to drop id
    df1 = drop_id_from_y(df1)
    df2 = drop_id_from_y(df2)
    
    #helper function to make class labels
    y_train, y_test, classes_dict = create_class_labels_for_y(df1, df2)
    
    return y_train, y_test, classes_dict        


def identifying_feature_types(df):
    """This function lists out the names of columns
    in a dataframe whose dtypes are objects as categorical
    features and the numeric types as numeric features
    
    Returns two lists of columns names first for numeric 
    and the second for categorical features
    """
    
    numeric_features = []
    categorical_features = []
    for name, dtype in zip(df.dtypes.index, df.dtypes):
        if str(dtype).startswith('ob'):
            categorical_features.append(name)
        else:
            numeric_features.append(name)
    
    #append the list with region and district code
    categorical_features.append('district_code')
    
    return numeric_features, categorical_features


def one_hot_encode_feature(df, name):
    """This funciton takes in a dataframe and a feature name and 
    One hot encodes the feature and adds it to the dataframe
    
    Returns transformed dataframe and the ohe object 
    used to transform the frame
    """
    
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    single_feature_df = df[[name]]
    ohe.fit(single_feature_df)
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    #returning ohe here so that it can be used to transform X_test later
    return df, ohe


def ohe_all_categorical_features(df):
    """This function takes in a dataframe, identifies the
    dtypes in the dataframe and uses the object dtypes to
    list out categorical columns
    
    Next it use OneHotEncoder to convert those Categorical 
    features
    
    Returns: the transformed dataframe and a dictionary 
    containing the ohe object that can be used later to 
    transform the testing dataset
    """
    
    
    #helper function to identify categorical feature names
    num_feats, cat_feats = identifying_feature_types(df)
    
    #reassuring the values in categorical features are str types
    for name in cat_feats:
        df[name] = df[name].astype(str)
    
    #use helper function in loop to transform dataframe
    encoders = {}
    
    for name in cat_feats:
        df, ohe = one_hot_encode_feature(df, name)
        encoders[name] = ohe
    
    return df, encoders


def one_hot_encode_test_features(df, name, ohe):
    """This funciton takes in the test dataframe, a feature name and 
    an ohe object and then One hot encodes the feature and adds
    it to the dataframe
    
    Returns the transformed test dataframe
    """
    
    single_feature_df = df[[name]]
    feature_array = ohe.transform(single_feature_df).toarray()
    ohe_df = pd.DataFrame(feature_array, columns=ohe.categories_[0], index=df.index)
    df = df.drop(name, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    
    return df


def ohe_train_and_test_features(X_train, X_test):
    
    X_train, encoders = ohe_all_categorical_features(X_train)
    
    X_test['district_code'] = X_test['district_code'].astype(str)
    
    for key in encoders:
        X_test = one_hot_encode_test_features(X_test, key, encoders[key])
    
    return X_train, X_test

{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
