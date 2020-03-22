# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:38:51 2020

@author: regal_000
"""

import joblib
import glob
import os
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Process the train Data as for the principle for test sets 
train_data = fetch_20newsgroups(subset='train',random_state=101)
x_train = train_data.data
y_train = train_data.target
labels = pd.DataFrame(train_data.target_names,columns = ["Category"])
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train) 
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
###################################################################
label_dic = {}
for key,value in enumerate(train_data.target_names):
    label_dic[key] = value


st.header("An NLP App Demo")
if st.checkbox("Show Categories"):
    st.table(labels)


csv_files = glob.glob("data/*")
file = st.selectbox("Choose a CSV file",csv_files)
data = pd.read_csv(file)
data_head = data.head(10)
st.dataframe(data_head)

st.header("Start Analysis")
data_list =[]
for line in range(len(data)):
    data_list.append(str(data.loc[line]))


st.subheader("BackEnd Processing")
x_test = count_vect.transform(data_list)
st.markdown("CountVectorizer's Shape")
st.write(x_test.shape)

x_test_tfidf = tfidf_transformer.transform(x_test)
st.markdown("Shape After TF-IDF")
st.write(x_test_tfidf.shape)


model_folder = glob.glob("model/*")
model_selector = st.selectbox("Select a Pretrained Model",model_folder,index = 2)
model = joblib.load(model_selector)
predictions = model.predict(x_test_tfidf)
labeled = []
for item in predictions:
    labeled.append(label_dic[item])
st.header("Predictions")
num = st.number_input("Show first X rows",value = 1,max_value = len(labeled))
st.table(labeled[:num])
st.markdown("Download Results")
if st.button("Download"):
    labeled = pd.DataFrame(labeled,columns = ["Predictions"])
    labeled = pd.concat([data,labeled],axis =1)
    labeled.to_csv("Predictions.csv")
    st.info("Download Successfully")