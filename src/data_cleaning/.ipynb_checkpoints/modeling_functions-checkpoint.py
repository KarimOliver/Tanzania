import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
from sklearn.metrics import recall_score, make_scorer

def class_2_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)[2]

def scorer():
    scorer = make_scorer(class_2_recall)
    return scorer

{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}
