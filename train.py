import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


x = pd.read_csv('train.csv')

y = 'quality' 


X_train = x.drop(y, axis=1)
y_train = x[y]


model = LogisticRegression(random_state=42, max_iter=5000)
print("Training Logistic Regression model...")


#model = RandomForestClassifier(random_state=42)
#print("Training Random Forest model...")

model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/model.joblib')
