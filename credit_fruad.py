import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Load the dataset
df = pd.read_csv('creditcard.csv')
# Display the first few rows of the dataset
print(df.head())
# Check for values in the 'Class' column
print(df["Class"].value_counts())
# Checking for missing values
print(df.isnull().sum())
# check the shapeof the dataset
print(df.shape)
#check the data types of the columns
print(df.dtypes)
# To check the columns name
print(df.columns)
# to access the specific column that is class 
print(df['Class'])
# Check uniqueness 
print(df.nunique())
# Apply StandardScaler to the 'Amount'and 'Time' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'] .values.reshape(-1, 1))
df.drop(['Time','Amount'], axis=1, inplace=True)
# Train-test split
X= df.drop('Class', axis=1)
y= df['Class']
X_test, X_train, y_test, y_train = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
# Model training with logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Model training with Random Forest to handle class imbalance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred= rf.predict(X_test)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
# Evaluate the matrix
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
#  Now ROC Curve and AUC
y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
#plt.show()
# feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
#plt.show()
# Now tuninig the model with GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'class_weight': ['balanced']
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best F1 Score:", grid.best_score_)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Visualizing the distribution of the 'Amount' column
plt.figure(figsize=(10, 6)) 
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount (scaled)')
plt.ylabel('Frequency')
plt.show()
# Visualizing the distribution of the 'Time' column 
plt.figure(figsize=(10, 6))
sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Distribution of Transaction Times')
plt.xlabel('Time (scaled)')
plt.ylabel('Frequency')
plt.show()
# Visualizing the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
# Visualizing the distribution of the 'Class' column
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='Set1')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()



