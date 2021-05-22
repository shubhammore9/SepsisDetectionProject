#import sepsis_mod1
import sepsis_mod2 as m2
import sepsis_mod3 as m3
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
import sys
import json
'''
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config",required=True,help="JSOn cong")
args= parser.parse_args()
jsonstring = unquote(args.config)'''

#inputValues = float(sys.argv[1]

'''xval =  json.dumps(json.JSONDecoder().decode(inputValues))
f= open("arg.txt","w+")
clst = type(inputValues)
f.write(xval[5])'''
pca=pickle.load(open('pca.pkl','rb'))
etc=pickle.load(open('etc.pkl','rb'))


def query_input(pca,etc):

    attribute_list=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
           'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
           'HospAdmTime', 'ICULOS']

    # In[243]:

    attribute_dict = dict.fromkeys(attribute_list , 0)

    attribute_dict['HR']=float(sys.argv[1])
    attribute_dict['O2Sat']=float(sys.argv[2])
    attribute_dict['Temp']=float(sys.argv[3])
    attribute_dict['SBP']=float(sys.argv[4])
    attribute_dict['MAP']=float(sys.argv[5])
    attribute_dict['DBP']=float(sys.argv[6])
    attribute_dict['Resp']=float(sys.argv[7])
    attribute_dict['FiO2']=float(sys.argv[8])
    attribute_dict['pH']=float(sys.argv[9])
    attribute_dict['SaO2']=float(sys.argv[10])
    attribute_dict['BUN']=float(sys.argv[11])
    attribute_dict['Calcium']=float(sys.argv[12])
    attribute_dict['Chloride']=float(sys.argv[13])
    attribute_dict['Glucose']=float(sys.argv[14])
    attribute_dict['Hgb']=float(sys.argv[15])
    attribute_dict['WBC']=float(sys.argv[16])
    attribute_dict['Age']=float(sys.argv[17])
    attribute_dict['Gender']=float(sys.argv[18])
    attribute_dict['Unit1']=float(sys.argv[19])
    attribute_dict['Unit2']=float(sys.argv[20])
    attribute_dict['HospAdmTime']=float(sys.argv[21])
    attribute_dict['ICULOS']=float(sys.argv[22])
    #attribute_dic

    # In[264]:


    #from sklearn.preprocessing import MinMaxScaler

    user_input=pd.DataFrame.from_dict(attribute_dict,orient='index')      #<class 'pandas.core.frame.DataFrame'>
    #print(type(user_input)) #dataframe

    scalar = preprocessing.MinMaxScaler()
    scaled_data=scalar.fit_transform(user_input)             #<class 'numpy.ndarray'>
    #print(scaled_data)
    #print(type(scaled_data))#numpyarray

    temp = pd.DataFrame(scaled_data,index=user_input.index,columns=user_input.columns)
    input_data=temp.transpose()                               #<class 'pandas.core.frame.DataFrame'>
    #print(input_data)
    #print(type(input_data)) #dataframe
    red_data_pca=pca.transform(input_data)                      #<class 'numpy.ndarray'>
    #print(type(red_data_pca)) #numpy array
    #print(red_data_pca.shape)
    #print(red_data_pca)
   
    #New_predict = etc.predict(red_data_pca)

    row=[[-0.82725606,  0.5108799 ,  0.18160533, -0.06484416,  0.07008959,
       -0.41600648, -0.00278161, -0.1908027 , -0.00762703,  0.18052529,
       -0.01246281, -0.01008171, -0.01948647,  0.141554  , -0.02347556,
        0.0532162 ,  0.06456134, -0.00229938,  0.00542272,  0.0013134 ]]
    New_predict = etc.predict(row)

    print(int(New_predict[0]))


    if(int(New_predict[0])):
        score,dictionary=m2.sirs_score(attribute_dict['Temp'], attribute_dict['HR'], attribute_dict['Resp'], attribute_dict['WBC'])
        if(score<=2):
            print(1)
        if(score==3):
            print(2)
        if(score==4):
            print(3)
        medication_dict=m3.medical_rec(dictionary)
        #print("Medical recommendation")
        print(json.dumps(medication_dict))
        #print(attribute_dict)
		


        
query_input(pca,etc)


