
### Text Display

Allows you pass almost anything into it and Streamlit can read it properly. (text, data, Matplotlib figures, Altair charts)

St.Write()


# Models
**GET DATA**
```python
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
```


## Model Making

**MODEL**
```python
X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    classifier1 = RandomForestClassifier(
        # hyperperameters here
    )
    classifier1.fit(X_train, y_train)
    y_pred = classifier1.predict(X_test)
```


## Model Import

**LOAD PICKEL**
```python
def pickel_load():  # load in saved model
    pickle_in = open("model/classifier.pkl", "rb")
    classifier = pickle.load(pickle_in)
    return classifier
```

**PREDICTIONS**
```python
classifier2 = pickel_load()
    if disp_col.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    disp_col.success("The output is {}".format(result))
```

# Customization
```python
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
```

```python
st.balloons()
```

