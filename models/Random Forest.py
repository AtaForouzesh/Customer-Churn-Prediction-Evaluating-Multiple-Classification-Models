import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("My Trainings\churn_data_large.csv")

x = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
# y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=5)
rf.fit(X_train, y_train)
y_ = rf.predict(X_test)

cfm = confusion_matrix(y_test, y_)
acs = accuracy_score(y_test, y_)
cr = classification_report(y_test, y_)
print(cfm, "\n", acs, "\n", cr)
