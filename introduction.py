from ast import Num
from email.mime import image
from optparse import Option
from turtle import color
import streamlit as st

st.title("My first streamlit")

from PIL import Image
st.subheader("Tis is a subheader")

# image= Image.open("")
# st.image(image, use_column_width=True)

st.write("writin a text")
st.markdown("THIS IS A MARKDOWN CELL")

st.success("Congratulation you ran the app successfully")

st.info("this is an information for you")



st.warning("be cautious")

st.error("OOPs you ran into an error, you need to rerun your app again or unistallS")

st.help(range)

import numpy as np
import pandas as pd

dataframe = np.random.rand(10,20)

st.dataframe(dataframe)
st.text("----"*100)

df = pd.DataFrame(np.random.rand(10,20), columns=("col %d" % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))
st.text("----"*100)

#Display chart
chart_data=pd.DataFrame(np.random.randn(20,3), columns=['a','b', 'c'])
st.line_chart(chart_data)

st.text("----"*100)

st.area_chart(chart_data)


chart_data=pd.DataFrame(np.random.randn(50,3), columns=['a','b', 'c'])
st.bar_chart(chart_data)

import matplotlib.pyplot as plt

arr = np.random.normal(1,1, size=100)
plt.hist(arr, bins=20)

st.pyplot()
st.text("----"*100)

# import plotly
# import plotlt.figure_factory as ff

# #Adding distplot

# x1= np.random.randn(200)-2
# x2 = np.random.randn(200)
# x3= np.random.randn(200)-2

# hist_data= [x1,x2, x3]
# group_labels = ['Group1', 'GROUP2', Group3]

# fig = ff.create_displot(hist_data.group_labels, bin_size= [.2, .25, .5])

# st.plotly_chart(fig.use_container_width=True)

st.text("----"*100)

# df=pd.DataFrame(np.random(100,2)/[50,50]+[37.76,-122.4], columns=["lat", "long"])

# st.map(df)

st.text("----"*100)

# Creating buttons
if st.button("say hello"):
    st.write()
else:
    st.write('why are you here')

st.text("----"*100)

genre = st.radio("what is your favourite genre?", ("Comedy", 'Drama', 'Documentary'))

if genre == 'Comedy':
    st.write("you like comedy")
elif genre == 'Drama':
    st.write("yeah Drama is cool")
else:
    st.write("welcome to Documentary")

st.text("----"*100)

# seelect button
option = st.selectbox("how was your night??", ("Fantastic", 'Awesome', 'so-so'))
st.write("You  said your night was:", option)

st.text("----"*100)
option = st.multiselect("how was your night??, you can select multiple answers", ("Fantastic", 'Awesome', 'so-so'))
st.write("You  said your night was:", option)

st.text("----"*100)

# slider
age = st.slider('How old are you?', 18,75)
st.write("Your age is:", age)

st.text("----"*100)

values = st.slider("Select a rane of values",0, 200,(15,18))
st.write("You selected a range between:", values)

# num =st.select_slider("select from ", 1,100)
# st.write("You selected",num)

number = st.number_input("input number")
st.write("THe number you picked was:", number)

st.text("----"*100)
st.text("----"*100)

# file uploader
upload_file = st.file_uploader("Chose a csv file", type='csv')
if upload_file is not None:
    data = pd.read_csv(upload_file)
    st.write(data)
    st.success("Successfully uploaded")
else:
    st.markdown("Please upload a CSV file")
    #st.error("The file you uploaded is empty, please upload a valid file")

# colour picker
color = st.color_picker("Pick your preferred color:", '#00f900')
st.write("This your color:", color)


# slide bar
st.text("----"*100)
st.text("----"*100)

add_sidebar = st.sidebar.selectbox("what is your favourite course?", ("Data science", 'Machine_learning', 'others','not sure'))

# proress bar
import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete+1)

st.text("----"*100)
st.text("----"*100)

with st.spinner("wait for it..."):
    time.sleep(5)
st.success("successful")
st.balloons()

