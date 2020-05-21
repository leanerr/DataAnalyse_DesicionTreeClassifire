#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[7]:


pima = pd.read_excel("./Train.xlsx", index_col=None, header=None)
pima.head()


# # lets delete missing values (0)
# داده های از دست رفته را با مد در هر ستون جایگزین میکنیم

# In[8]:


from sklearn.impute import SimpleImputer 
imr = SimpleImputer(missing_values=0, strategy='most_frequent')#Stategies:median(ordinal),most_frequent(nominal)
imr = imr.fit(pima)
imputed_data = imr.transform(pima.values)
imputed_data


# # translating new array to new Dataset by name pima

# In[9]:


pima= pd.DataFrame(data=imputed_data, index=None)
pima


# we choose the col[16] of this dataset as target
# and lets make new dataset by name data to use 15 cols and target that are joined together

# In[10]:


#Feature selection
#split dataset in features and target variable
feature_cols = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 ]
X = pima[feature_cols] # Features
y = pima[16] # Target variable
data= X.join(y)
data.head()


# # lets train with 70% of our data , then test with 30%

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# # feature selection : its better to do the calculate by choosing the best features(cols)

# featureimportance is the Function that could tell us how much  important is each col or feature 

# In[2]:


def featureimportance(node):
    nodesum = sum(node.values())
    percents = {c:v/nodesum for c,v in node.items()}
    return nodesum, percents


# but lets calculate the importance of each col in a professional style

# In[12]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
feat_labels = data.columns[:15]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1) 
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))


# we can show the importances to deliver the point much better

# In[13]:


import matplotlib.pyplot as plt
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)                                    
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# so lets select the best features 

# In[105]:


#Feature selection again
#split dataset in features and target variable agian
feature_cols2 = [0, 1, 2, 3 , 4]
X1 = pima[feature_cols2] # Features
y1 = pima[16] # Target variable
data= X1.join(y1)
data.head()


# training again

# In[106]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=1) # 70% training and 30% test


# # using entropy : without final feature selection : DecisionTreeClassifier
# 

# entropyscore Fucntion is showing us the algorithm of calculation entropy score

# In[16]:


def entropyscore(node):
    nodesum, percents = calcpercent(node)
    score = round(sum([-i*log(i,2) for i in percents.values()]), 3)
    return score


# lets calculate Accuracy of Model without final feature selection and using entropy

# In[128]:


#Building Decision Tree Model without final feature selection
# Create Decision Tree classifer object
dd = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
dd = dd.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dd.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # using gini : without final feature selection : DecisionTreeClassifier

# giniscore Fucntion is showing us the algorithm of calculation gini score

# In[ ]:


def giniscore(node):
    nodesum, percents = calcpercent(node)
    score = round(1 - sum([i**2 for i in percents.values()]), 3)
    return score


# In[49]:


#Building Decision Tree Model without final feature selection
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[130]:


from sklearn import tree
tree.plot_tree(clf) 


# # using entropy : with final feature selection : DecisionTreeClassifier

# In[117]:


#Building Decision Tree Model with final feature selection
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X1_train,y1_train)

#Predict the response for test dataset
y1_pred = clf.predict(X1_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y1_test, y1_pred))


# # returning tree for entropy

# In[118]:


from sklearn import tree

tree.plot_tree(clf) 


# # using gini :with  final feature selection : DecisionTreeClassifier

# In[114]:


#Building Decision Tree Model with final feature selection
# Create Decision Tree classifer object
dd = DecisionTreeClassifier(criterion="gini", max_depth=4)

# Train Decision Tree Classifer
dd = dd.fit(X1_train,y1_train)

#Predict the response for test dataset
y_pred = dd.predict(X1_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y1_test, y1_pred))


# In[115]:


from sklearn import tree

tree.plot_tree(dd) 

