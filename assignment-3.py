import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC  
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv("sonar.all-data")  

#dividing data into attributes and lables
x = dataset.iloc[:,0:59].values   
y = dataset.iloc[:,60].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)  

# Training RandomForest
classifier = RandomForestClassifier(n_estimators=200,random_state=1)
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test) 

print("Random Forest Classifier Results : ")
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print("Accuracy")  
print(accuracy_score(y_test, y_pred)*100)  

# paramter tuning for random forest
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

tree = RandomForestClassifier()
classifier_cv = RandomizedSearchCV(tree, param_grid, cv =5)
classifier_cv.fit(X_train, y_train)  
print("Tuned Decision Tree Parameters: {}".format(classifier_cv.best_params_))
print("Best score is {}".format(classifier_cv.best_score_))


# Training with GiniIndex for Decision Tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state=1, max_depth = 3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)  #performing training
y_pred_gini = clf_gini.predict(X_test) 

# DecisionTree Results
print("Decision Tree Classifier Results using Gini Index: ")
print(confusion_matrix(y_test,y_pred_gini))  
print(classification_report(y_test,y_pred_gini)) 
print("Accuracy") 
print(accuracy_score(y_test, y_pred_gini)*100)  

#Training with Entropy for Decision Tree
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 1, max_depth = 3, min_samples_leaf = 5)   
clf_entropy.fit(X_train, y_train) 
y_pred_entropy = clf_entropy.predict(X_test) 

#DecisionTree Results
print("Decision Tree Classifier Results using Entropy: ")
print(confusion_matrix(y_test,y_pred_entropy))  
print(classification_report(y_test,y_pred_entropy)) 
print("Accuracy") 
print(accuracy_score(y_test, y_pred_entropy)*100)

# parameter tuning for decision tree
param_dist = {"max_depth": randint(1, 32), "max_features": randint(1, 9), "min_samples_leaf": randint(1, 9), "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X_train, y_train)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


#SVC
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test)  

#SVC Results
print("SVC Results: ")
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("Accuracy") 
print(accuracy_score(y_test, y_pred)*100)

# parameter tuning for svc
parameters = [{'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C':[1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'poly']}]
svclassifier = GridSearchCV(SVC(random_state=777), parameters, cv=5, scoring='accuracy', n_jobs = -1)
svclassifier.fit(X_train, y_train)
print("Tuned Decision Tree Parameters: {}".format(svclassifier.best_params_))
print("Best score is {}".format(svclassifier.best_score_))


