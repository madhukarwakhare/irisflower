import streamlit as st
import sklearn 
import joblib


st.title("MACHINE LEARNING -IRIS")

sepal_length = st.slider('Enter sepal_length', 0.1, 7.9, 2.0) 
sepal_width = st.slider('Enter sepal_width', 0.1, 7.9, 2.0)
petal_length = st.slider('Enter petal_length', 0.1, 7.9, 2.0)
petal_width = st.slider('Enter petal_width', 0.1, 7.9, 2.0)


from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(x,y)

y=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
output=iris.target_names[y[0]]
st.write("iris class is",output)
