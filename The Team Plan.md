### 1. Download & Open Streamlit (ERIK)

### 2. Basics
- Header
- Compartments
- Columns

### 3. Charts
- Table/Df
- Graphs
   - Stretch: Plotly

### 4. Import Model 
- How to Import - making my model inside streamlit, but 
- Get user input
- Display output

### 5. Wrap it up
- Caching
   1. make it a function
   2. call the function
   3. add `@st.cache`
   ```python
   @st.cache
   def get_data():
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
- changing background
   ```python
   st.markdown(
    """
    <style>
    .main {
        background-color: #81b29a;
    }
    </style>
    """,
    unsafe_allow_html=True,
   )
   ```
