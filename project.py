import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from PIL import Image

# Set title
st.title("Project")

# image = Image.open("")
# st.image(image, use_column_width=True)

# set subtitile
st.write("""
# A simple Data App with Streamlit
""")

st.write("""
### Let's explore different classifiers and datsets
""")

dataset_name = st.sidebar.selectbox("Select dataset", ("Breast Cancer", 'Iris', 'Wine'))

classifier_name = st.sidebar.selectbox("Select Classifier", ("SVM", 'KNN',"RandomForestClassifier"))

def get_dataset(name):
    data =None
    if name == "Iris":
        data= datasets.load_iris()
    elif name== 'Wine':
        data =datasets.load_wine()
    # elif name == "Boston Housing":
    #     data = datasets.load_boston()
    else:
        data = datasets.load_breast_cancer()
    x= data.data
    y = data.target
    return x, y


x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your dataset is :', x.shape)
st.write('Unique target variables:', len(np.unique(y)))

fig =plt.figure()
sns.boxplot(data=x, orient = 'h')

#remove warning
# st.pyplot()st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)

plt.hist(x)
st.pyplot()

#Building our algorithm

def add_parameter(name_of_clf):
    params= dict()
    if name_of_clf=="SVM":
        C = st.sidebar.slider("C", 0.01, 15.0)
        params['C']= C   
    elif name_of_clf =="RandomForestClassifier":
        pass
    else:
        name_of_clf== "KNN"
        k = st.sidebar.slider('k',1,15)
        params['k'] = k
    return params
    
    
params = add_parameter(classifier_name)

# set random seed
randoms = st.sidebar.slider("rand", 1, 70)

# Accessing our classifier
def get_classifier(name_of_clf, params):
    clf =None
    if name_of_clf =="SVM":
        clf = SVC(C = params['C'])
    elif name_of_clf =="RandomForestClassifier":
        clf = RandomForestClassifier()
    else:
        clf=KNeighborsClassifier(n_neighbors=params['k'])
    return clf

clf =get_classifier(classifier_name, params)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=randoms)

clf.fit(x_train, y_train)

y_preds = clf.predict(x_test)
st.write(y_preds)

accuracy = accuracy_score(y_test, y_preds)
st.write("classifier name:", classifier_name)
st.write("Accuracy for your model is:", accuracy)