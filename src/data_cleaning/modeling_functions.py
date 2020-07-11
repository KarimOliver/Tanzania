import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
from sklearn.metrics import recall_score, make_scorer

def class_2_recall(y_true, y_pred):
    """In this function we create a customized scorer
    to return the recall score for class 2
    """
    
    return recall_score(y_true, y_pred, average=None)[2]

def scorer():
    """This function makes use of make_scorer from
    sklearn.metrics to customize our scoring for cross validation
    
    we use a helper function that returns the recall for class 2 
    alone, this is important since cross_val_score can be used
    only if the scoring returns a single score for each folds.
    
    Since we are using classification for 3 classes,
    getting a single recall score needs customization
    """
    
    #helper function to return recall score for class 2
    scorer = make_scorer(class_2_recall)
    
    return scorer

{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
