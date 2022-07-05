#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


### importing  dataset 

data = pd.read_csv('train.csv.csv')


# In[3]:


## checking first five rows in dataset
data.head()


# In[4]:


### Check number of rows and columns in dataset


# In[5]:


data.shape


# In[6]:


### Check the information of the dataset


# In[7]:


data.info()


# In[8]:


### checking the null values in dataset


# In[9]:


data.isnull().sum()


# In[10]:


### statistical measures


# In[11]:


data.describe()


# # EDA(Exploratory Data Analysis)

# In[12]:


### Finding number Risk_Flag or Not Risk_Flag

data['Risk_Flag'].value_counts()


# In[13]:


#### Making a countplot of 'Risk_Flag'


# In[14]:


plt.figure(figsize=(8,8))
sns.countplot('Risk_Flag',data=data)


# In[15]:


### Find number of states in 'Risk_flag'


# In[16]:


data['STATE'].value_counts()


# In[17]:


### Making countplot of 'States' with respect to 'Risk_Flag'


# In[18]:


plt.figure(figsize=(30,15))
sns.countplot(x='STATE',hue='Risk_Flag',data=data)


# In[19]:


### Checking column of 'House_Ownership' 


# In[20]:


data['House_Ownership'].value_counts()


# In[21]:


#### Making countplot of 'House_Ownership' with respect to 'Risk_Flag'


# In[22]:


plt.figure(figsize=(15,15))
sns.countplot(x='House_Ownership',hue='Risk_Flag',data=data)


# In[23]:


### check the column of 'Car_Ownership'


# In[24]:


data['Car_Ownership'].value_counts()


# In[25]:


### Making countplot of 'Car_Ownership' with respect to 'Risk_Flag'


# In[26]:


plt.figure(figsize=(8,8))
sns.countplot(x='Car_Ownership',hue='Risk_Flag',data=data)


# In[27]:


### Check the column of 'Married/Single'


# In[28]:


data['Married/Single'].value_counts()


# In[29]:


### Making countplot of 'Married/Singal' with respect to 'Risk_Flag'


# In[30]:


plt.figure(figsize=(8,8))
sns.countplot(x='Married/Single',hue='Risk_Flag',data=data)
plt.xlabel='Married/Single'
plt.ylabel='Count'
plt.show()


# In[31]:


## check the columns of 'Experience'


# In[32]:


data['Experience'].value_counts()


# In[33]:


## Making countplot of 'Experience' with respect to 'Risk_Flag'


# In[34]:


plt.figure(figsize=(10,10))
sns.countplot(x='Experience',hue='Risk_Flag',data=data)
plt.xlabel='Experience'
plt.ylabel='Count'
plt.show()


# In[35]:


## check the columns of 'Age'


# In[36]:


data['Age'].value_counts()


# In[37]:


## Making countplot of 'Age' with respect to 'Risk_Flag'


# In[38]:


plt.figure(figsize=(20,10))
sns.countplot(x='Age',hue='Risk_Flag',data=data)
plt.xlabel='Age'
plt.ylabel='Count'
plt.show()


# In[39]:


### Converting categorical values into Numerical values 


# In[40]:


### Converting categorical Columns
data.replace({'Married/Single':{'single':0,'married':1},'House_Ownership':{'rented':0,'owned':1,'norent_noown':2},
             'Car_Ownership':{'no':0,'yes':1}},inplace=True)


# In[41]:


data


# In[42]:


## Drop the Id,profession,CITY,STATE from the dataset


# In[43]:


df = data.drop(['Id','Profession','CITY','STATE'],axis=1)


# In[44]:


df


# In[45]:


### DF(Dataset) split in X(Feature variable) and y(target variable) 


# In[46]:


x = df.iloc[:,0:8].values

y = df.iloc[:,8].values


# In[47]:


x


# In[48]:


y


# In[49]:


### Split X and Y into train and test sets


# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[51]:


### print the shape of x_train and x_test


# In[52]:


print(x.shape,x_train.shape,x_test.shape)


# In[53]:


### print the shape of y_train and y_test


# In[54]:


print(y.shape,y_train.shape,y_test.shape)


# # Training the Random Forest Calssification Model on the Training set

# In[55]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 2)
classifier.fit(x_train, y_train)


# # Finding Accuracy

# In[56]:


x_train_pred = classifier.predict(x_train)


# In[57]:


### Print X_train_pred

x_train_pred


# In[58]:


## Probability of getting output as 1 - Risk_Flag

y_pred1 = classifier.predict_proba(x_test)[:,1]


# In[59]:


from sklearn.metrics import accuracy_score


# In[60]:


training_data_accuracy = accuracy_score(y_train,x_train_pred)
print('Accuracy score of training data :',training_data_accuracy)


# In[61]:


x_test_pred = classifier.predict(x_test)


# In[62]:


training_data_accuracy = accuracy_score(y_test,x_test_pred)
print('Accuracy score of training data :',training_data_accuracy)


# #  The training-set accuracy score is 0.93160 while the test-set accuracy to be 0.8968 . These two values are quite comparable .So , there is no question of overfitting

# # Metrics for model evalution
#    
# #   Confusion Matrix¶

# In[63]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, x_test_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[64]:


# visualize confusion matrix with seaborn heatmap


# In[65]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Negative:0', 'Actual Positive:1'], 
                                 index=['Predict Negative:0', 'Predict Positive:1'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Classification metrics

# In[66]:


from sklearn.metrics import classification_report

print(classification_report(y_test, x_test_pred))


# In[67]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[68]:


# print classification accuracy


# In[69]:


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# # Classification error 

# In[70]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# # Precision

# In[71]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# # Recall

# In[72]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# # Specifity

# In[73]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# # F1-Score

# In[74]:


# plot ROC Curve

# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Risk_Flag classifier')

# plt.xlabel('False Positive Rate (1 - Specificity)')

# plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[75]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# # K fold cross validation

# In[76]:


### Applaying 5-fold cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier,x_train,y_train,cv=5,scoring='accuracy')

print('Cross-Validation scores:{}'.format(scores))


# In[ ]:




