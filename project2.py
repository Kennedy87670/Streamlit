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
                st.write(sns.heatmap(df.corr(), vmax=1,square=True,annot=True, cmap='viridis'))
                st.pyplot()
        
    
    # elif option == 'model':
    #     ....
    
    # elif option == 'About us':


main()