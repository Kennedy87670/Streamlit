from ast import Param
from optparse import Option
import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
matplotlib.use('Agg')

from PIL import Image

#Set title

st.title('Project_2')
# image = Image.open('')
# st.image(image,use_column_width=True)


def main():
    activities=["EDA", 'Visualization', 'model', "About us"]
    option= st.sidebar.selectbox("Select option", activities)


# dealing with eda
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx,' 'txt', 'json'])
        st.success('Data successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Display shape'):
                st.write(df.shape)
            if st.checkbox('Display columns'):
                st.write(df.columns)
            if st.checkbox('Select multiple columns'):
                selected_columns = st.multiselect("Select preferred columns:", df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('display summary'):
                st.write(df.describe().T)
            if st.checkbox("Display data types"):
                st.write(df.dtypes)
            if st.checkbox("Display missing values"):
                st.write(df.isnull().sum())
            if st.checkbox('Display Correclation of various data columns'):
                st.write(df.corr())

 # dealing with visualization       
    elif option == "Visualization":
        st.subheader(" Data Visualization")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx,' 'txt', 'json'])
        st.success('Data successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple columns to plot'):
                selected_columns= st.multiselect('Select your preferred columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sns.heatmap(df1.corr(), vmax=1,square=True,annot=True, cmap='viridis'))
                st.pyplot()
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1, diag_kind="kde"))
                st.pyplot()
            if st.checkbox("Display Pie Chart"):
                all_column =df.columns.to_list()
                pie_columns = st.select_slider("Select column to display", all_column)
                piechart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(piechart)
                st.pyplot()
        
# dealing with model part    
    elif option == 'model':
        st.subheader('Model building')

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx,' 'txt', 'json'])
        st.success('Data successfully loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Select Multiple coulmns"):
                new_data = st.multiselect("Select your preferred columns:", df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # creating x and y
                #X = df1[:-1]
                #y = df1[-1:]
                X = df1.iloc[:,0:-1]
                y = df1.iloc[:,-1]
            
            seed = st.sidebar.slider("Seed", 1, 100)

            classifier_name = st.sidebar.selectbox("Select your preferred classifier", ('KNN', 'SVM', 'LogisticRegression', 'Naive_bayes', 'Decision tree', 'RandomForestClassifier'))

            def add_parameter(name_of_clf):
                params = dict()
                if name_of_clf == 'SVM':
                    C = st.sidebar.slider("C", 0.01, 15.0)
                    params['C'] =C
                if name_of_clf == "KNN":
                    K = st.sidebar.slider('K', 1, 15)
                    params['K']=K
                    return params
            # calling the fuction

            params =add_parameter(classifier_name)

            # define a function for classifier
            def get_classifier(name_of_clf, params):
                clf =None
                if name_of_clf == 'SVM':
                    clf = SVC(C = params['C'])
                elif name_of_clf == 'KNN':
                    clf=KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf == 'LogisticRegression()':
                    clf = LogisticRegression()
                    pass
                elif name_of_clf =="RandomForestClassifier":
                    clf = RandomForestClassifier()
                elif name_of_clf == 'Naive_bayes':
                    clf =GaussianNB()
                elif name_of_clf == 'Decision tree':
                    clf =DecisionTreeClassifier()
                else:
                    st.warning("Select your choice of a classifier")

                return clf

            clf = get_classifier(classifier_name, params)

                        
            X_train,X_test,y_train,y_test=train_test_split(X, y,  test_size=0.2, random_state=seed)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)
            st.write(y_preds)

            accuracy = accuracy_score(y_test, y_preds)
            st.write("classifier name:", classifier_name)
            st.write("Accuracy for your model is:", accuracy)


#DELING WITH THE ABOUT US PAGE
    elif option=='About us':
        st.markdown('This is an interactive web page for our ML project, feel feel free to use it. This dataset is fetched from the UCI Machine learning repository. The analysis in here is to demonstrate how we can present our wok to our stakeholders in an interractive way by building a web app for our machine learning algorithms using different dataset.')
        st.balloons()
	# 	..............


if __name__ == '__main__':
    main() 


