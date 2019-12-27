# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:51:33 2019

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_excel('Wine Data set.xlsx')
dataset.isnull().any()
dataset.describe()
dataset.dtypes
sns.countplot(x="quality", data=dataset)

reviews = []
for i in dataset['quality']:
 if i >= 1 and i <= 5 :
  reviews.append('0')
 elif i >= 6 and i <= 10:
  reviews.append('1')
dataset['reviews'] = reviews

sns.pairplot(dataset, vars=dataset.columns[:-1])

fig, ax = plt.subplots(figsize=(10,10))
corr = dataset.corr()
# plot the heatmap
sns.heatmap(corr,annot=True,xticklabels=corr.columns,yticklabels=corr.columns)

m=dataset.iloc[:,:-2].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
m=sc.fit_transform(m)
m=pd.DataFrame(m)
from sklearn.decomposition import PCA
pca=PCA()
m=pca.fit_transform(m)
pca.components_[0]
res=pca.explained_variance_ratio_*100
np.cumsum(pca.explained_variance_ratio_*100)
scores=pd.Series(pca.components_[0])
scores.abs().sort_values(ascending=False)
var=pca.components_[0]
plt.bar(x=range(1,len(var)+1),height=res)

#Index 6,5,1,3,4,9,0,8,2,-->95%  7,11,10 Not required
#Feature 7,6,2,4,5,10,1,9,3 Taken as Important  1,2,3,4,5,6,7,9,10
# Remove 8,11,12 () Density, alcohol, quality

#Model Building  - Logistic Regression
dataset1=dataset.drop(['density'],axis=1)
h=dataset1.iloc[:,:-3].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(h)
#Encoding ( 0-Red & 1- White)

y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
classifier.predict_proba(x_test)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix (y_test,y_pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
print(accuracy*100)

#Accuracy = 98% 
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




Decision Tree Classifier 



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_excel('Wine Data set.xlsx')

dataset1=dataset.drop(['density'],axis=1)
h=dataset1.iloc[:,:-3].values    #Feature drop after PCA
y=dataset.iloc[:,-1].values

x=pd.DataFrame(h)

#Standard SCALER not required for decision Tree

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix (y_test,y_pred)
print(cm)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
print(accuracy*100)

from sklearn.model_selection import cross_val_score
acc=cross_val_score(classifier,x_train,y_train,cv=12)
acc.mean()
acc.std()


98% Accuracy with CV =12

feature_cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulphur dioxide','total sulpur dioxide','pH','sulphates']
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data=StringIO()
export_graphviz(classifier,out_file=dot_data,filled=True,rounded=True,special_characters=True,
                feature_names=feature_cols,class_names=['Red','White'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Wine Classification.png')
Image(graph.create_png())

import pandas as pd 
feature_imp=pd.Series(classifier.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important features')
plt.show()


#Plot Confusion Matrix 
from sklearn.metrics import confusion_matrix
import numpy as np
mat=confusion_matrix(y_pred,y_test)
names=np.unique(y_pred)
import seaborn as sns
sns.set()
sns.heatmap(mat,square=True,annot=True,fmt='d',cbar=False,xticklabels=names,yticklabels=names)
import matplotlib.pyplot as plt
plt.xlabel('Actual')
plt.ylabel('Predicted')

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#EVALUATION METRICS
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

Accuracy: 0.9853846153846154
Precision: 0.9938900203665988
Recall: 0.9868554095045501

#Inference 
'''Well, you got a classification rate of 98%, considered as good accuracy.
Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In your prediction case, when your model predicted Wines are going to be classified as red, that wines were really red have 99% of the time.
Recall: If there are wines  which have been classified as red in the test set and your model can identify it 98% of the time.'''


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
 clf.fit(X_train, y_train)
 print("Accuracy on training set:")
 print(clf.score(X_train, y_train))
 print("Accuracy on testing set:")
 print(clf.score(X_test, y_test))
 y_pred = clf.predict(X_test)
 print("Classification Report:")
 print(metrics.classification_report(y_test,y_pred))

# Compute confusion matrix

#SVM
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
classifier = SVC(gamma='auto', kernel='linear')
train_and_evaluate(classifier, x_train, x_test, y_train, y_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




'''
RBF Kernel 
Accuracy on testing set: 
0.9453846153846154

Linear 
Accuracy on testing set:
0.9869230769230769'''



