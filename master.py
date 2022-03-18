import os

os.system('cmd /k "pip install -r requirements.txt"')

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Graphing packages
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# stuff for machine
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- SETUP --- #

## Containers
header = st.container()
dataset = st.container()
exploration = st.container()
model_making = st.container()
model_import = st.container()

## Functions(*skip*)
@st.cache
def get_data():  # load in the data
    # the data
    iris = datasets.load_iris()
    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    return data


# @st.cache
def pickel_load():  # load in saved model
    pickle_in = open("model/classifier.pkl", "rb")
    classifier = pickle.load(pickle_in)
    return classifier


def prediction(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier2.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )
    print(prediction)
    return prediction


## Customize(*skip*)
# palette: https://coolors.co/palette/f4f1de-e07a5f-3d405b-81b29a-f2cc8f
st.markdown(
    """
<style>
.main {
    background-color: #cad2c5;
}
</style>
    """,
    unsafe_allow_html=True,
)


# --- BODY ---
# header
with header:
    st.title("Streamlit & Machine Learning")
    st.write(
        """
        *Following this [tutorial](https://www.youtube.com/watch?v=-IM3531b1XU&list=PLM8lYG2MzHmRpyrk9_j9FW0HiMwD9jSl5)*
        """
    )

# dataset
with dataset:
    df = pd.read_csv("data/iris.csv")

    st.title("CheatSheet")

    st.markdown(
        "There are a variety of ways to display information inside of streamlit"
    )

    st.header("Display Text")

    st.code("st.markdown()")
    st.markdown("This is in markdown")

    st.code("st.latex()")
    st.latex(
        r""" \underbrace{\overbrace{\ b_0 \ }^\text{y-intercept} + \overbrace{b_1}^\text{slope} X_i \ }_\text{estimated regression relation}
    """
    )

    st.code("st.write()")
    st.write(
        "Allows you to display all types of data and information. Streamlit just knows..."
    )

    st.header("Display Data")

    st.markdown("Dataframes")

    st.markdown("Using Magic")
    st.code("df")
    df

    st.code("st.dataframe()")
    st.dataframe(df)

    st.code("st.table()")

    if st.checkbox("Show table"):

        st.table(df)

    st.code("st.write()")

    st.write(df)

    st.markdown("Charts")

    st.code("st.write()")
    sepal = (
        alt.Chart(data=df, title="Flower Sepal Measurements")
        .encode(
            x="sepal length (cm)",
            y="sepal width (cm)",
            color="Type",
            tooltip=["sepal length (cm)", "sepal width (cm)"],
        )
        .mark_circle()
        .interactive()
    )
    st.write(sepal)

    st.header("Why choose dataframe/table over Write?")

    st.markdown("1. dataframe/table allows for the data to be added or replaced")
    st.markdown(
        "2. dataframe/table have various arguments that can be used to customize table"
    )

    if "df" not in st.session_state:
        # st.session_state.df =df
        st.session_state.df = pd.DataFrame(
            columns=[
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
                "Variety",
            ]
        )

    st.subheader("Add Record")

    num_new_rows = st.sidebar.number_input("Add Rows", 1, 50)
    ncolumns = st.session_state.df.shape[1]  # col count

    rw = -1

    with st.form(key="add form", clear_on_submit=True):
        cols = st.columns(ncolumns)
        rwdta = []

        for i in range(ncolumns):
            rwdta.append(cols[i].text_input(st.session_state.df.columns[i]))

        # you can insert code for a list comprehension here to change the data (rwdta)
        # values into integer / float, if required

        if st.form_submit_button("Add"):
            if st.session_state.df.shape[0] == num_new_rows:
                st.error("Add row limit reached. Cant add any more records..")
            else:
                rw = st.session_state.df.shape[0] + 1
                st.info(f"Row: {rw} / {num_new_rows} added")
                st.session_state.df.loc[rw] = rwdta

                if st.session_state.df.shape[0] == num_new_rows:
                    st.error("Add row limit reached...")
    button = st.button("Delete Table")

    st.dataframe(st.session_state.df)

    st.code(
        "element = st.dataframe(st.session_state.df) \nelement.add_rows(st.session_state.df)"
    )

    element = st.dataframe(st.session_state.df)
    element.add_rows(st.session_state.df)

    if button:
        for key in st.session_state.keys():
            del st.session_state[key]

    st.header("Session State")

    """
    - Session current app, when a new app is open it is a new session
    - State is what is used to store the current values on the back end
    - Session State is how the session and state communicate to one another, so you can store previous run throughs of the app

    

    Think of Run state as a python dictionary that operates in a key value pair.

    """
    st.subheader("accessing keys")
    st.code("for the_key in st.session_state.keys(): \n st.write(the_key)")
    for the_key in st.session_state.keys():
        st.write(the_key)

    st.subheader("accessing values")
    st.code("for the_value in st.session_state.values(): \n st.write(the_value)")
    for the_value in st.session_state.values():
        st.write(the_value)

    st.subheader("accessing pairs")
    st.code("for the_item in st.session_state.items(): \n st.write(the_item)")
    for the_item in st.session_state.items():
        st.write(the_item)

    st.markdown(
        "[Cheatsheet](https://docs.streamlit.io/library/cheatsheet)",
        unsafe_allow_html=True,
    )


# exploration
with exploration:

    data = get_data()

    if st.checkbox("Show Something"):
        st.write("*SOMETHING HERE*")
    st.write("Built in charts")
    st.line_chart(data)
    st.write(
        "The built in charts are not very customizable. They infer based off of what the dataframe contains."
    )

    chart = (
        alt.Chart(data)
        .encode(
            alt.X("petal length", axis=alt.Axis(title="Petal Length")),
            alt.Y("petal width", axis=alt.Axis(title="Petal Width")),
            color=alt.Color("species", title="Species"),
        )
        .mark_circle()
        .configure_axis(labelFontSize=18, titleFontSize=18)
        .configure_title(fontSize=20)
        .configure_legend(titleFontSize=18, labelFontSize=18)
    )
    st.write("Vega Charts (includes Altair)")
    st.altair_chart(chart)
    st.write("We are given a lot more control when we use other graphing libraries")

    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(x="sepal length", y="sepal width", hue="species", data=data)
    st.write("Matplotlib (includes seaborn)")
    st.pyplot(fig)

    st.write(
        "There are many more packages you can use, including: deck.gl, plotly, bokeh, pydeck, and graphviz"
    )


# model_making
with model_making:
    st.header("Model - Make & Tune")
    sel_col, disp_col = st.columns(2)

    ### inputs
    max_depth = sel_col.slider(
        "What should be the max_depth of the model?",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )
    n_estimators = sel_col.selectbox(
        "How many trees should there be?", options=[100, 200, 300], index=1
    )
    min_samples_split = sel_col.radio(
        "What should be the min_samples_split? ", (2, 3, 4, 5)
    )

    ### machine
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier1 = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    classifier1.fit(X_train, y_train)
    y_pred = classifier1.predict(X_test)

    ### display
    disp_col.subheader("Mean Absolute Error of the model is: ")
    disp_col.write(mean_absolute_error(y_test, y_pred))

    disp_col.subheader("Mean Squared Error of the model is: ")
    disp_col.write(mean_squared_error(y_test, y_pred))

    disp_col.subheader("R Squared Score of the model is: ")
    disp_col.write(r2_score(y_test, y_pred))


# model_import
with model_import:
    st.header("Model - Import & Predict")
    sel_col, disp_col = st.columns(2)

    ### inputs
    sepal_length = sel_col.number_input(
        "Sepal Length", value=(data["sepal length"].mean())
    )
    sepal_width = sel_col.number_input(
        "Sepal Width", value=(data["sepal width"].mean())
    )
    petal_length = sel_col.number_input(
        "Petal Length", value=(data["petal length"].mean())
    )
    petal_width = sel_col.number_input(
        "Petal Width", value=(data["petal width"].mean())
    )
    result = ""

    ### predictions
    classifier2 = pickel_load()
    if disp_col.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
        st.balloons()
    disp_col.success("The output is {}".format(result))
