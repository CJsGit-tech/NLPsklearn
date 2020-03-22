# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:56:20 2020

@author: regal_000
"""
import joblib
from sklearn.datasets import fetch_20newsgroups

train_data = fetch_20newsgroups(subset='train',
                                  shuffle=True,random_state=101)

test_data = fetch_20newsgroups(subset ="test",random_state =101)


##############################################################
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(train_data.data) 
print("CountVectorizer")
print(x_train_counts.shape)  


from sklearn.feature_extraction.text import TfidfTransformer
# use_idf = False since we only have counts no weights
tfidf_transformer = TfidfTransformer(use_idf = False)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print("TFIDF Matrix Shape")
print(x_train_tfidf.shape)

x_test = count_vect.transform(test_data.data)
print("CountVectorizer")
print(x_test.shape)  

x_test_tfidf = tfidf_transformer.transform(x_test)
print("TFIDF Matrix Shape")
print(x_test_tfidf.shape)
#####################################################
model = joblib.load("model/SVC.sav")
pred = model.predict(x_test_tfidf)

from sklearn.metrics import classification_report,confusion_matrix
classification = classification_report(pred,test_data.target,target_names = test_data.target_names)
print(classification)

