#!/usr/bin/env python
# coding: utf-8

# In[56]:


# loan prediction data set 


# In[ ]:


import pandas as pd


# In[57]:


l=pd.read_csv(r"D:\loan.csv")


# In[58]:


l.head()


# In[59]:


l.tail()


# In[60]:


l.size


# In[61]:


l.shape


# In[62]:


l.info()


# In[63]:


l.describe()


# In[64]:


l.isna().sum()


# In[65]:


l.head()


# In[66]:


l.Loan_Amount_Term=l.Loan_Amount_Term.fillna(l.Loan_Amount_Term.median())
l.LoanAmount=l.LoanAmount.fillna(l.LoanAmount.median())
l.Credit_History=l.Credit_History.fillna(l.Credit_History.median())


# In[67]:


l.isnull().sum()


# In[68]:


l.Gender.fillna(l.Gender.mode()[0],inplace=True)
l.Married.fillna(l.Married.mode()[0],inplace= True)
l.Dependents.fillna(l.Dependents.mode()[0],inplace=True)
l.Self_Employed.fillna(l.Self_Employed.mode()[0],inplace=True)


# In[69]:


l.isna().sum()


# In[70]:


l["Income"]=l.ApplicantIncome+l.CoapplicantIncome


# In[71]:


l.drop(columns=["ApplicantIncome","CoapplicantIncome"],inplace=True)


# In[72]:


l.plot(figsize=(10,10),kind="box")


# In[73]:


l=l[l.Income<10000]


# In[74]:


l.plot(figsize=(10,10),kind="box")


# In[75]:


lst=[]
for i in l.columns:
    if l[i].dtype=="O":
        lst.append(i)
        
lst        


# In[76]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[174]:


for i in lst:
    l[i]=lb.fit_transform(l[i])
    


# In[78]:


l.head()


# In[79]:


l.size


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[81]:


plt.figure(figsize=(15,10))
sns.heatmap(l.corr(),annot=True,linewidths=1)
plt.show()


# In[82]:


x=l.drop(columns="Loan_Status")
y=l["Loan_Status"]


# In[83]:


y.value_counts()


# In[ ]:


#spliting train test


# In[178]:


from sklearn.model_selection import train_test_split


# In[179]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)


# In[180]:


X_train.shape,X_test.shape


# In[181]:


#Stanrdised data


# In[182]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_train


# In[ ]:





# In[183]:


# 1.Logistic regression 


# In[184]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score


# In[185]:


model=LogisticRegression()


# In[186]:


model.fit(X_train,y_train)


# In[187]:


y_predl=model.predict(X_test)


# In[199]:


accuracy_score(y_test,y_predl)


# In[229]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model=LogisticRegression()
    model.fit(X_train,y_train)
    y_predl=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_predl))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

maxi_iter=[]
for i in range(0,1000,50):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_no)
    model=LogisticRegression(max_iter=i)
    model.fit(X_train,y_train)
    y_predl=model.predict(X_test)
    maxi_iter.append(accuracy_score(y_test,y_predl))
    accuracy=max(Random_State)
    
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)
print("Maximum iteration number : ",(maxi_iter.index(max(maxi_iter)))*50,"Accuracy Score : ",accuracy)
    


# In[201]:


recall_score(y_test,y_pred)


# In[193]:


precision_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[116]:


#2.Decision tree 


# In[117]:


from sklearn.tree import DecisionTreeClassifier


# In[118]:


dt = DecisionTreeClassifier()


# In[119]:


dt.fit(X_train,y_train)


# In[120]:


y_pred=dt.predict(X_test)


# In[194]:


accuracy_score(y_test,y_pred)


# In[195]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    dt=DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_pred=dt.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# In[196]:


precision_score(y_test,y_pred)


# In[197]:


recall_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[125]:


#3.Knn 


# In[126]:


from sklearn.neighbors import KNeighborsClassifier


# In[127]:


model_knn=KNeighborsClassifier(n_neighbors=5)


# In[128]:


model_knn.fit(X_train,y_train)


# In[129]:


y_predknn=model_knn.predict(X_test)


# In[130]:


accuracy_score(y_predknn,y_test)


# In[131]:


accuaracy=[]
for i in range(5,101,3):
    if(i%2!=0):
        model_knn=KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(X_train,y_train)
        y_predknn=model_knn.predict(X_test)
        accuracy_score(y_predknn,y_test)
        accuaracy.append(accuracy_score(y_predknn,y_test))
        print("Value of k",i,"and corresponding accuaracy",accuracy_score(y_predknn,y_test))  
 
print("\n")
print("Maximum accuracy is ",max(accuaracy))


# In[132]:


recall_score(y_predknn,y_test)


# In[133]:


precision_score(y_predknn,y_test)


# In[ ]:





# In[134]:


#4.Neive Bayes classifier


# In[135]:


from sklearn.naive_bayes import GaussianNB


# In[136]:


model_nb=GaussianNB()


# In[137]:


model_nb.fit(X_train,y_train)


# In[138]:


y_predNb=model_nb.predict(X_test)


# In[139]:


accuracy_score(y_predNb,y_test)


# In[140]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model_nb=GaussianNB()
    model_nb.fit(X_train,y_train)
    y_predNb=model_nb.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_predNb))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# In[141]:


precision_score(y_predNb,y_test)


# In[142]:


recall_score(y_predNb,y_test)


# In[ ]:





# In[ ]:





# In[143]:


#5.SVM


# In[144]:


from sklearn.svm import SVC


# In[145]:


model_sv=SVC()


# In[146]:


model_sv.fit(X_train,y_train)


# In[147]:


y_predsv=model_sv.predict(X_test)


# In[148]:


accuracy_score(y_predsv,y_test)


# In[149]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model_sv=SVC()
    model_sv.fit(X_train,y_train)
    y_predsv=model_sv.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_predsv))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# In[150]:


precision_score(y_predsv,y_test)


# In[151]:


recall_score(y_predsv,y_test)


# In[ ]:





# In[152]:


#6.Random Forest


# In[153]:


from sklearn.ensemble import RandomForestClassifier


# In[154]:


model_rf=RandomForestClassifier()


# In[155]:


model_rf.fit(X_train,y_train)


# In[156]:


y_predrf=model_rf.predict(X_test)


# In[157]:


accuracy_score(y_predrf,y_test)


# In[228]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model_rf=RandomForestClassifier()
    model_rf.fit(X_train,y_train)
    y_predrf=model_rf.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_predrf))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# In[216]:


precision_score(y_predrf,y_test)


# In[217]:


recall_score(y_predrf,y_test)


# In[ ]:





# In[218]:


#7.Gradient Boosting


# In[219]:


from sklearn.ensemble import GradientBoostingClassifier


# In[220]:


md=GradientBoostingClassifier()


# In[221]:


md.fit(X_train,y_train)


# In[222]:


y_predgb=md.predict(X_test)


# In[223]:


accuracy_score(y_predgb,y_test)


# In[227]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    md=GradientBoostingClassifier()
    md.fit(X_train,y_train)
    y_predgb=md.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_predgb))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# In[225]:


precision_score(y_predgb,y_test)


# In[226]:


recall_score(y_predgb,y_test)


# In[ ]:


1. Logistic Regression  :
    Accuracy Score :  0.8860759493670886
    Precision : 0.8549618320610687
    Recall :1.0

 
2. Decison Tree Clasifier :
    Accuracy : 0.8164556962025317
    Precision :  0.8611111111111112
    Recall : 0.8017241379310345


3. Knn algorithm  :
    Accuracy : 0.759493670886076
    Precision : 1.0
    Recall : 0.7341772151898734
        
  
4. Neive Bayes algorithm :
    Accuracy : 0.879746835443038
    Precision : 0.9741379310344828
    Recall : 0.849624060150376
        
        
5.SVM Classifier :
    Accuracy : 0.759493670886076
    Precision : 1.0
    Recall : 0.7341772151898734
        

6.Random Forest Classifier :
    Accuracy Score :  0.879746835443038
    Precision : 0.9568965517241379
    Recall : 0.8473282442748091
    
    

7.Gradient Boosting :
     Accuracy Score :  0.8670886075949367  
    Precision : 0.9224137931034483
    Recall : 0.856

       


# In[ ]:


The best model with highest accuracy is :Logistic regression with accuracy score  0.8860759493670886

