# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:34:47 2018

@author: anmol
"""

from sklearn import model_selection, preprocessing, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics




from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 


import pandas as pd
import numpy as np
import os
import re

import pickle

dataset_list, corpus = [],[]

directory_path = 'C:/Users/anmol/Documents/my_lab/Python_lab/dataset/'
resume_data_path = 'C:/Users/anmol/Documents/my_lab/Python_lab/dataset/'
bbc_data_path = 'C:/Users/anmol/Documents/my_lab/Python_lab/bbc_100.csv'
data_type_path = 'C:/Users/anmol/Documents/my_lab/type.csv'

# Taking each file in designated directory
for i in os.listdir(directory_path):
    # Only taking those files whose attribute is .txt                                         
    if i.endswith('.txt'):
        # Opening file and reading each line
        with open( resume_data_path + i, 'r', 
                  encoding = 'ISO-8859-1') as f:
            
            dataset_list.append(''.join(f.readlines()))
            
# Creates a DataFrame with corpus column
dataset =pd.DataFrame(dataset_list, columns = ['corpus'])

# Create dataFrame for BBC news set
bbc_dataset = pd.DataFrame()

# Read csv file and put data in corpus header of DataFrame
bbc_dataset['corpus'] = pd.read_csv(bbc_data_path)

# Merge dataset and bbc_dataset
dataset = dataset.append(bbc_dataset, ignore_index = True)

# Read the class file and import in DataFrame 'type' column
dataset['type'] = pd.read_csv(data_type_path) 

#==============================================================================
#
#==============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='ISO-8859-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(dataset.corpus).toarray()
labels = dataset.type

#==============================================================================
#
#==============================================================================

model = MultinomialNB()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataset.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)

y_pred = model.predict(X_test)

accuracy_percent = metrics.accuracy_score(y_test, y_pred)
print('Accuracy (Metrics):', accuracy_percent)



filename = 'doc_classifier_model.sav'
# Dumping the fitted model
pickle.dump(model,open(filename, 'wb'))

