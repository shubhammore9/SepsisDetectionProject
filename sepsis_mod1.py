#!/usr/bin/env python
# coding: utf-8

# In[265]:

import pickle
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from collections import Counter
import io
#from google.colab import files   
#uploaded = files.upload()
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier


# In[152]:

def train_model():

    data = pd.read_csv('sepsis.csv')

    # In[153]:

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(data)
    x= imputer.transform(data)

    # In[154]:


    #from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)


    # In[171]:


    x = data.loc[:, data.columns!= 'SepsisLabel']
    y = data.loc[:, data.columns == 'SepsisLabel']
    #type(x)


    # In[221]:


    #from sklearn.decomposition import PCA

    pca = PCA(n_components=20)

    x_pca = pca.fit_transform(x)
    #scores=pd.DataFrame(x_pca)
    #print(scores)
    explained_variance = pca.explained_variance_ratio_
    #print(pd.DataFrame(pca.components_,columns=x.columns))
    #explained_variance
    #s=pca.components_
    #s.shape
    #x_pca.shape
    pickle.dump(pca,open('pca.pkl','wb'))



    # In[175]:


    #from sklearn.model_selection import train_test_split, StratifiedKFold
    X_train, X_test, y_train, y_test = train_test_split(x_pca,y,test_size=0.25, random_state = 62)


    # In[177]:


    #!pip install -U imbalanced-learn
    #import imblearn
    #from imblearn.over_sampling import SMOTE


    # In[273]:


    #%%time
    kf = StratifiedKFold(n_splits=4)
    cross_val_f1_score_lst = []
    cross_val_accuracy_lst = []
    cross_val_recall_lst = []
    cross_val_precision_lst = []


    for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
        # keeping validation set apart and oversampling in each iteration using smote 
        train, validation = X_train[train_index_ls], X_train[validation_index_ls]
        target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
        sm = SMOTE(random_state=62)
        X_train_res, y_train_res = sm.fit_resample(train, target_train)
        #print (X_train_res.shape, y_train_res.shape)
        
       #ExtraTreeClassifier
        #from sklearn.ensemble import ExtraTreesClassifier
        etc = ExtraTreesClassifier( n_estimators=1000, criterion="entropy", max_features="auto", min_samples_leaf=1, min_samples_split=5)
        #pickle.dump(etc,open('etc.pkl','wb'))
        
        # fit the model on the whole dataset
        etc.fit(X_train_res, y_train_res)
        
        #predict on the train data
        etc_predict = etc.predict(validation)
        
        '''
        from sklearn.metrics import accuracy_score,f1_score,recall_score, precision_score,confusion_matrix
        cross_val_recall_lst.append(recall_score(target_val, etc_predict))
        cross_val_accuracy_lst.append(accuracy_score(target_val, etc_predict))
        cross_val_precision_lst.append(precision_score(target_val, etc_predict))
        cross_val_f1_score_lst.append(f1_score(target_val, etc_predict))
    print ('Cross validated accuracy: {}'.format(np.mean(cross_val_accuracy_lst)))
    print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall_lst)))
    print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision_lst)))
    print ('Cross validated f1_score: {}'.format(np.mean(cross_val_f1_score_lst)))
    '''
        


    # In[248]:



    #predict on the test data
    etc_predict_test = etc.predict(X_test)
    #print(X_test)
    #print(etc_predict_test)
    #print(type(X_test))
    #print(len(X_test))

    '''
    from sklearn.metrics import accuracy_score,f1_score,recall_score, precision_score,confusion_matrix
    print("accuracy:",accuracy_score(etc_predict_test,y_test))
    print("recall score:",recall_score(etc_predict_test,y_test))
    print("precision score: ", precision_score(etc_predict_test,y_test))
    print("f1 score: ", f1_score(etc_predict_test,y_test))
    print('train-set confusion matrix:\n', confusion_matrix(etc_predict_test,y_test))
    etc_list=[accuracy_score(etc_predict_test,y_test),recall_score(etc_predict_test,y_test),precision_score(etc_predict_test,y_test),f1_score(etc_predict_test,y_test)]
    print(etc_list)
    '''
    pickle.dump(etc,open('etc.pkl','wb'))


 
train_model()

