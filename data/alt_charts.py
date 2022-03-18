import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

df = pd.read_csv("data/iris.csv")

st.table(df.sample(frac=.3).style.highlight_max(axis=0))

petal = alt.Chart(data=df, title = "Flower Petal Measurements").encode(
    x = 'petal length (cm)',
    y = 'petal width (cm)',
    color = 'Type',
    tooltip = ['petal length (cm)', 'petal width (cm)']
).mark_circle().interactive()

sepal = alt.Chart(data=df, title='Flower Sepal Measurements').encode(
    x = 'sepal length (cm)',
    y = 'sepal width (cm)',
    color = 'Type',
    tooltip = ['sepal length (cm)','sepal width (cm)']
).mark_circle().interactive()

st.altair_chart(petal)

st.altair_chart(sepal)

