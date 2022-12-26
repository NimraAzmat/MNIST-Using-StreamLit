import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import cv2
from PIL import Image, ImageEnhance
#@title Splitting the dataset into train and test set 70-30 
from sklearn.model_selection import train_test_split
# Import datasets, classifiers and performance metrics
from sklearn import  svm, metrics
import matplotlib.pyplot as plt

#creating containers that will hold up the related information of the project
header=st.container()
dataset=st.container()
features=st.container()
modelTraining=st.container()
test_prediction=st.container()
modelPrediction = st.container()


#title of the project
with header:
    st.header('''**Hand Written Digit Prediction**''')
    ## Project Description
    st.caption("## **About Project:**")
    st.write('''This project will Predict the Numbers depending on the different image that the
user will provide.''')

#About dataset
with dataset:
    st.header("**Dataset:**")
    st.write('''The Digits dataset from Scikit-Learn Library consist of different features that are presented as Dictionary:

    - Dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])''')

# Data Dictionary
with features:
    #Block for Imorting the dataset
    st.text("Below is the dataset loading steps:")
    digit = load_digits()
    #Block for Analyzing the features
    st.write('''**Data Set Characteristics:**

    :Number of Instances: 1797
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998''')
    df =pd.DataFrame(digit.data,columns=digit.feature_names)
    #Block for Visualization of the dataset
    st.write('**First 10 Colums of the dataset:**') 
    df['Lables']=digit.target
    st.write(df.head(5))
    st.subheader('Digits Distribution Plot')
    dist_data=pd.DataFrame(df['Lables'].value_counts())
    st.bar_chart(dist_data)
    # flatten the images
    n_samples = len(digit.images)
    data = digit.images.reshape((n_samples, -1))
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    for ax, image, label in zip(axes, digit.images, digit.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    st.subheader("Image Model Training Results:")
    st.pyplot(fig)

with modelTraining:
    #Block for Train test spliting 
    X_train, X_test, y_train, y_test = train_test_split(data, digit.target, test_size=0.5, shuffle=False)

    #Block for Model Creation
    # Create a classifier: a support vector classifier
    model = svm.SVC(gamma=0.001)

    # Learn the digits on the train subset
    model.fit(X_train, y_train)

with test_prediction:
    st.header("Prediction of the Model:")
    predicted = model.predict(X_test)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label,prediction in zip(axes, X_test,y_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Pred: {prediction} : True: {label}")
    st.subheader("Model Prediction:")
    st.pyplot(fig)