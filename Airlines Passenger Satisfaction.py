#!/usr/bin/env python
# coding: utf-8

# # Airlines Passenger Satisfiction Analysis using Machine Learning

# In[1]:


#Important Library and function imported here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#importing dataset here
data = pd.read_csv('passenger.csv')


# In[4]:


#dataset print from head
data.head()


# In[5]:


#datset print from tail
data.tail()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe()


# ## Data Visualization Using Histogram

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.hist(bins = 50,figsize=(20,30))


# ## Label Encoding for Nominal Data in our Dataset

# In[10]:


l_encoder = LabelEncoder()
data['Gender']= l_encoder.fit_transform(data['Gender'])
data['Customer Type']= l_encoder.fit_transform(data['Customer Type'])
data['Type of Travel']= l_encoder.fit_transform(data['Type of Travel'])
data['Class']= l_encoder.fit_transform(data['Class'])
data['satisfaction']= l_encoder.fit_transform(data['satisfaction'])


# #### After Label Encoding data head print again here

# In[11]:


data.head()


# #### After Label Encoding data tail print again here

# In[12]:


data.tail()


# #### Plotting Histogram Again

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.hist(bins=50,figsize=(20,30))


# ## Scatter Ploting Vizualization

# In[14]:


sns.scatterplot(x='Flight Distance',y='Food and drink',hue = 'satisfaction' ,data = data)


# ## Countplot

# In[15]:


sns.countplot(data['Gender'])


# In[16]:


sns.countplot(data['Type of Travel'])


# In[17]:


sns.countplot(data['Flight Distance'])


# In[18]:


sns.countplot(data['Class'])


# In[19]:


sns.countplot(data['Age'])


# In[20]:


sns.countplot(data['satisfaction'])


# ##### Note:  Not Making all attribute countplot. But following this command we can make all attribute countplot

# ## Heat Map Crating

# In[21]:


plt.figure(figsize=(10,5))
sns.heatmap(data.corr(),annot =True)


# In[22]:


attributes = ["Gender","Customer Type","Age","Type of Travel","Class","Flight Distance"]
scatter_matrix(data[attributes],figsize = (12,8))


# ## Missing Value Finding and Handeling

# In[23]:


#Checking for missing value
data.isnull().sum()


# #### Note: we have found 83 missing value on Arrival Delay in Minutes. Now we need to handle this missing value by using Mean() replace formula

# In[24]:


#Missing value handeling code
data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].replace(np.NaN,data['Arrival Delay in Minutes'].mean())


# In[25]:


#Checking for Missing Remain or not!
data.isnull().sum()


# #### Now we have no missing value in our data set

# In[26]:


# Count plot creating for those Attribute

sns.countplot(data['Arrival Delay in Minutes'])


# ## Spliting Dataset

# In[27]:


# Spliting Data set in x and y 

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]


# In[28]:


x.head()


# In[29]:


y.head()


# In[30]:


x


# In[31]:


y


# In[32]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[33]:


x_train


# In[34]:


x_test


# In[35]:


y_train


# In[36]:


y_test


# ### Feature Scaling

# In[37]:


#Standard Scaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.fit_transform(x_test)


# In[38]:


x_train


# In[39]:


x_test


# ## Model Fitting 

# ### Logistic Regression

# In[40]:


lr_model = LogisticRegression()


# In[41]:


lr_model.fit(x_train,y_train)


# In[42]:


predict_LR = lr_model.predict(x_test)


# In[43]:


predict_LR


# ### Report Analysis for Logistic Regression

# In[44]:


print("Logistic Regression Report")
print("---------------------------")
print("Confusion Matrix: \n",confusion_matrix(y_test,predict_LR))


# In[45]:


print("Accuracy: ",accuracy_score(y_test,predict_LR)*100)


# In[46]:


print('Classification Report: \n',classification_report(y_test,predict_LR))


# ##### So after applying Logistic Regression in our Dataset we found accuracy 86.80.

# #### Logistic Regression AUC Score Calculation

# In[47]:


auc_score = metrics.roc_auc_score(y_test,predict_LR)


# In[48]:


auc_score


# #### Logistic Regression ROC Score Calculation

# In[79]:


fpr1,tpr1,threshold = roc_curve(y_test,predict_LR) 
roc_auc1= auc(fpr1,tpr1)
roc_auc1


# #### ROC Curve plotting for Logistic Regression

# In[80]:


plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.plot(fpr1,tpr1, label = 'Logistic Regression Area = %0.2f' % roc_auc1)
plt.legend(loc = 'lower right')
plt.title('Customer Satisfaction')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# ## Support Vector Machine

# In[51]:


sm = SVC(probability=True)


# In[52]:


sm.fit(x_train,y_train)


# In[53]:


y_predict_sm = sm.predict(x_test)


# In[54]:


y_predict_sm


# ### Report Analysis for SVM

# In[55]:


print('Support Vector Machine Report')
print('------------------------------')
print('Confusion Matrix: \n',confusion_matrix(y_test,y_predict_sm))


# In[56]:


print('Accuracy of SVM: ',accuracy_score(y_test,y_predict_sm)*100)


# In[57]:


print('Classification Report: \n',classification_report(y_test,y_predict_sm))


# ##### Now Applying Support Vector Machine (SVM) model on our Dataset we found 94.44 Accuracy Rate. Which is better then Logistic Regression.

# #### SVM AUC Score Calculation

# In[58]:


auc_score = metrics.roc_auc_score(y_test,y_predict_sm)


# In[59]:


auc_score


# #### SVM ROC Score Calculation

# In[77]:


fpr2,tpr2,threshold = roc_curve(y_test,y_predict_sm) 
roc_auc2 = auc(fpr1,tpr2)
roc_auc2


# #### SVM ROC Curve Ploting
# 

# In[78]:


plt.plot([0,1],[0,1],color='green',linestyle='--')
plt.plot(fpr1,tpr1, label = 'SVM Area = %0.2f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.title('Customer Satisfaction')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# ## Decision Tree

# In[63]:


dec_tree = DecisionTreeClassifier(random_state = 0)


# In[64]:


dec_tree.fit(x_train,y_train)


# In[65]:


dec_tree_predict= dec_tree.predict(x_test)


# In[66]:


dec_tree_predict


# ### Report Analysis for Decision Tree

# In[67]:


print('Report Analysis for DecisionTreeClassifier:')
print('--------------------------------------------')
print('Confusion Matrics: \n',confusion_matrix(y_test,dec_tree_predict))


# In[68]:


print('Accuracy of DecisionTreeClassifier: ',accuracy_score(y_test,dec_tree_predict)*100)


# In[69]:


print('Classification Report of DecisionTreeClassifier: \n',classification_report(y_test,dec_tree_predict))


# #### AUC Score calculation for DecisionTreeClassifier

# In[70]:


auc_score = metrics.roc_auc_score(y_test,dec_tree_predict)


# In[71]:


auc_score


# #### DecisionTreeClassifier ROC Score Calculation

# In[75]:


fpr3,tpr3,threshold = roc_curve(y_test,y_predict_sm) 
roc_auc3 = auc(fpr3,tpr3)
roc_auc3


# #### DecisionTreeClassifier ROC ploting

# In[76]:


plt.plot([0,1],[0,1],color='purple',linestyle='--')
plt.plot(fpr3,tpr3, label = 'DecisionTreeClassifier Area = %0.2f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.title('Customer Satisfaction')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# In[82]:


plt.plot(fpr1, tpr1, label='Logistic Model (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, label='SVC Model (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, label='DecisionTreeClassifier (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiverrating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




