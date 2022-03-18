#%%
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

# %%
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

X.columns = iris.feature_names
y.columns = ['Type']
y.replace([0,1,2], iris.target_names, inplace=True)

df = pd.concat([X,y],axis=1)
df.to_csv('iris.csv',index=False)
# %%
