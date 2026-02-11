# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and create the employee dataset using pandas DataFrame.
2. Split the dataset into input features (X) and target variable (Churn), then divide into training and testing sets.
3. Train a Decision Tree Classifier using the training data and predict churn on the test data.
4. Evaluate the model performance using accuracy, confusion matrix, and visualize the decision tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Denesh Raj Balaji Rao
RegisterNumber:  25005647
*/
```

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
```

```
df = pd.DataFrame(data)
```

```
X = df[['satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company']]

y = df['left']
```

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

```
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

model.fit(X_train, y_train)
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,              
    min_samples_split=50,     
    min_samples_leaf=20,      
    random_state=42
)

model.fit(X_train, y_train)
```

```
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
```

```
plt.figure(figsize=(14, 7))

plot_tree(model,
          feature_names=X.columns,
          class_names=['Stayed', 'Left'],
          filled=True,
          rounded=True,
          fontsize=8)

plt.title("Decision Tree for Employee Churn Prediction")
plt.show()
```

```
new_emp = [[0.35, 0.55, 3, 160, 3]]

prediction = model.predict(new_emp)

if prediction[0] == 1:
    print("Employee is likely to LEAVE")
else:
    print("Employee is likely to STAY")

```


## Output:
<img width="842" height="848" alt="image" src="https://github.com/user-attachments/assets/af0d9f0c-a0bf-49df-8d8b-9ff583b0a734" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
