import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

data = pd.read_csv('df.csv')
data = data.drop(['mandate', 'nationality_final'], axis=1)