# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:46:39 2020

@author: regal_000
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_20newsgroups

train_data = fetch_20newsgroups(subset='train',
                                  shuffle=True,random_state=101)

test_data = fetch_20newsgroups(subset ="test",random_state =101)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(train_data.data) 
print("CountVectorizer")
print(x_train_counts.shape)  

x_test = count_vect.transform(test_data.data)
print("CountVectorizer")
print(x_test.shape)  



from sklearn.feature_extraction.text import TfidfTransformer
# use_idf = False since we only have counts no weights
tfidf_transformer = TfidfTransformer().fit(x_train_counts)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print("TFIDF Matrix Shape")
print(x_train_tfidf.shape)


x_test_tfidf = tfidf_transformer.transform(x_test)
print("TFIDF Matrix Shape")
print(x_test_tfidf.shape)



""" Naive Bayes Classifier"""
from sklearn.naive_bayes import MultinomialNB
naive= MultinomialNB().fit(x_train_tfidf,train_data.target)
train_pred = naive.predict(x_train_tfidf)
train_score = np.mean(train_data.target == train_pred)
print("Train Score for Naive Bayes")
print(train_score)
joblib.dump(naive,"NaiveBayes.sav")

test_pred = naive.predict(x_test_tfidf)
test_score = np.mean(test_data.target == test_pred)
print("Test Score for Naive Bayes")
print(test_score)
print("Shape", x_test_tfidf.shape)




""" SGD """
from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier().fit(x_train_tfidf,train_data.target)
train_pred = SGD.predict(x_train_tfidf)
train_score = np.mean(train_data.target == train_pred)
print("Train Score for SGD")
print(train_score)
joblib.dump(SGD,"SGD.sav")

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(weights = "distance").fit(x_train_tfidf,train_data.target)
train_pred = knn.predict(x_train_tfidf)
train_score = np.mean(train_data.target == train_pred)
print("Train Score for KNN")
print(train_score)
joblib.dump(knn,"KNN.sav")

""" MLP """
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (128,64,32,10),
        max_iter = 10,early_stopping = True,verbose = True,
        random_state = 101).fit(x_train_tfidf,train_data.target)
train_pred = mlp.predict(x_train_tfidf)
train_score = np.mean(train_data.target == train_pred)
print("Train Score for MLP")
print(train_score)
print("Activation Function")
print(mlp.out_activation_)
joblib.dump(mlp,"MLP.sav")


""" Linear SVC """
from sklearn.svm import LinearSVC
svc = LinearSVC(random_state = 101).fit(x_train_tfidf,train_data.target)
train_pred = knn.predict(x_train_tfidf)
train_score = np.mean(train_data.target == train_pred)
print("Train Score for SVC")
print(train_score)
joblib.dump(svc,"SVC.sav")

######################################################################

