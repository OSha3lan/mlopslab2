import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

test_df = pd.read_csv('test.csv')
model = joblib.load('models/model.joblib')



TARGET = 'quality' 

if TARGET not in test_df.columns:
    print(f"Error: Target column '{TARGET}' not found in the test data.")
    print(f"Available columns: {list(test_df.columns)}")
    exit()

X_test = test_df.drop(TARGET, axis=1)
y_test = test_df[TARGET]


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")


metrics = {'accuracy': accuracy}


with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to 'metrics.json'")


cm = confusion_matrix(y_test, predictions, labels=model.classes_)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')


plt.savefig('confusion_matrix.png')

print("Confusion matrix plot saved to 'confusion_matrix.png'")