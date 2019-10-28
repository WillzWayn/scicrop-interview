#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import modules
import numpy as np # Numeric operations
import pandas as pd # Data manipulation
import pickle

import warnings
warnings.filterwarnings("ignore")

## Location to models in api.

path='WEB MachineLearning model deployed with Flask/static/modelos/'


# In[ ]:


# Read the initial dataset
apy = pd.read_csv('apy.csv')
"""
Context
Historical data of Indian agricultural
production on various location acquired from the Indian government web page.
https://data.gov.in""";


# In[ ]:


# A Look in the dataframe
apy.head()


# In[ ]:


# Check Shape
apy.shape


# In[ ]:


# Check Variable types
apy.dtypes

# Production tem problema !


# In[ ]:


# check for null values
apy.isnull().sum()


# In[ ]:


apy[apy['Production'] == '='].head(2)


# In[ ]:


#Years in apy dataset
#sorted((apy.Crop_Year.value_counts().index))


# In[ ]:


gdp = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_247793.csv', skiprows=4)


# In[ ]:


gdp.head(3)


# In[ ]:


# Locate India
gdp.loc[gdp['Country Name'] == 'India']


# In[ ]:


# Select only 1997-2015
start_year = gdp.columns.get_loc('1997')
end_year = gdp.columns.get_loc('2015')
# India is row 107
gdp_9715 = gdp.iloc[107, start_year:end_year+1]


# In[ ]:


# Create a Dataframe for gdp
gdp_9715_idx = gdp_9715.index[:]
gdp_dict = {'Year': gdp_9715_idx, 'GDP': gdp_9715}
gdp_df = pd.DataFrame(gdp_dict)
gdp_df = gdp_df.reset_index(drop=True)
gdp_df.head()


# In[ ]:


# Check if Crop_Year have all values from 1997 to 2015
apy_crop_year = apy['Crop_Year']
apy_crop_year.unique()


# In[ ]:


# Check Datatypes
gdp_df.dtypes


# In[ ]:


# Convert year to int
gdp_df['Year'] = gdp_df['Year'].astype(int)


# In[ ]:


# Map all GDP values to the respective year
rename_dict = gdp_df.set_index('Year').to_dict()['GDP']
apy_crop_year = apy_crop_year.replace(rename_dict)
# Create the definitive GDP Dataframe
apy_crop_year_dict = {'GDP': apy_crop_year}
gdp_final = pd.DataFrame(apy_crop_year_dict)


# In[ ]:


# See the data type
gdp_final['GDP'].dtype


# In[ ]:


# Concatenate the DataFrames
india_crop_gdp_1997_2015 = pd.concat([apy, gdp_final], axis=1, sort=False)


# In[ ]:


# A look into DataFrame
india_crop_gdp_1997_2015.head()


# In[ ]:


# Save as .csv
india_crop_gdp_1997_2015.to_csv('india_crop_gdp_1997_2015.csv')


# In[ ]:


india_crop_gdp_1997_2015= pd.read_csv('india_crop_gdp_1997_2015.csv')
india_crop_gdp_1997_2015.columns


# In[ ]:


# Label Enconding
from sklearn import preprocessing
RelationItens ={}
for f in india_crop_gdp_1997_2015.columns:
    if f == 'Production':
        continue
    if india_crop_gdp_1997_2015[f].dtype =='object': 
        le = preprocessing.LabelEncoder()
        itens = (sorted(list(india_crop_gdp_1997_2015[f].value_counts().index)))
        le.fit(list(india_crop_gdp_1997_2015[f].values))
        itens_edited = le.transform(itens)
        RelationItens[f]={b:a for a,b in zip (itens,itens_edited)}
        india_crop_gdp_1997_2015[f] = le.transform(list(india_crop_gdp_1997_2015[f].values))
#del RelationItens['Production']
RelationItens.keys()


# In[ ]:


#for i in RelationItens['Season']:
#    print ('        <option value="{}">{}</option>'.format(i[1],i[0]))


# # Separação do data set
# 
# ## Usaremos 2 datasets para aplicar a pesquisa! O que acontece é que a coluna de Produção apresenta problemas em alguns valores. Todavia algoritimo 1 não será interferido por esse problema !
# 
# 
# ### Algorithm 1. Problema de Classificação ! usaremos o dataset original.
# 
# ### Algorithm 2. Problema de Regressão ! Usaremos uma variação do dataset. Onde retiramos os problemas na coluna de Production!

# # Model 1 - Classification Problem (Decision Tree Classifier)
# 
# - Algoritmo 1:  Objetivo: descobrir a CROP (Cultivo). A partir de 3 dados aleatórios (AREA, STATE, DISTRICT) inseridos pelo usuário, seja calculado o resultado de CROP

# In[ ]:


# Select variables (Features)
X1 = india_crop_gdp_1997_2015[['Area', 'District_Name', 'Season']]
y1 = india_crop_gdp_1997_2015['Crop']


# In[ ]:


# Split Train and Test
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, 
                                                    test_size = 0.15,
                                                   random_state = 42)

model1 = DecisionTreeClassifier(random_state=42)
model1.fit(x1_train, y1_train)


# In[ ]:


# Predict and calculate scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

y1_pred = model1.predict(x1_test)
acc = accuracy_score(y1_test, y1_pred)
pre = precision_score(y1_test, y1_pred, average='micro')
rec = recall_score(y1_test, y1_pred, average='micro')
f1 = f1_score(y1_test, y1_pred, average='micro')


print('Accuracy: ', acc)
print('Precision: ', pre)
print('Recall: ', rec)
print('F1-Score: ', f1)


# In[ ]:


#------   Saving the model with pickle -----------------
# Define o nome do arquivo em disco que irá guardar o nosso modelo
filename = 'model_1_FindCrop.sav'
# salva o modelo no disco
pickle.dump(model1, open(path+filename, 'wb'))


# # Model 2 - Regression Linear !
# 
# - Algoritmo 2: Objetivo: descobrir a PRODUCTION (Produção agrícola de um cultivo). A partir de 3 dados aleatórios (AREA, CROP e GDP) inseridos pelo usuário, seja calculado o resultado de PRODUCTION

# In[ ]:


#Retiro os itens com erro na produção
# Existem produção que possuem o valor =
list1 = (list(india_crop_gdp_1997_2015['Crop'].value_counts().index))
df = india_crop_gdp_1997_2015[india_crop_gdp_1997_2015['Production'] != '=']
# Transformo produção em float
df['Production']=df['Production'].astype(float)

#Pego apenas valores de produção com Valores maiores que zero.
#Pois algum erro deve ter acontecido e encheu nosso dataframe de zero.
#Prejudicando as predições.

df2 = df.query('Production > 0')

list2 = list(df2['Crop'].value_counts().index)
list2=sorted(list2)

diferenca = [x for x in list1 if x not in list2]
#[x for x in list1 if ]

#lista_final
print( 'Os elementos', sorted(diferenca), 'não possuem dados suficientes para realizar a projeção !')


# In[ ]:


#india_crop_gdp_1997_2015.query("Crop == 1")


# ### Visualizando os dados:

# ### Separando os dados por grupos de Cultivo. Faz sentido analizar a produção de banana com banana, maça com maça, etc.

# In[ ]:


a=list(df2.groupby('Crop'))
len(a)


# ### Parece existir uma linearidade dos dados para Crops individuais !

# In[ ]:


RelationItensBackUp = RelationItens
for k in diferenca:
    RelationItens['Crop'].pop(k, None)


# In[ ]:


cropList = [RelationItens['Crop'][i] for i in list2]


# ### Vamos Aplicar uma regressão linear nos dados desagrupados!

# In[ ]:


from sklearn.metrics import r2_score

def productionML(x2,y2,crop):
    '''
    Função que tem objetivo de aplicar uma Regressão Linear nos dados
    e salvar o arquivo como o pickle'''
    # Split Train and Test
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, 
                                                            test_size = 0.30,
                                                           random_state = 42)
    
    from sklearn.linear_model import LinearRegression

    #from sklearn.linear_model import LogisticRegression
    model2 = LinearRegression()

    model2.fit(x2_train, y2_train)
    y2_pred = model2.predict(x2_test)
    r2 = r2_score(y2_test, y2_pred)
    #- R-squared (Coefficient of determination) represents the coefficient of how well the values fit compared to the original values.
    # The value from 0 to 1 interpreted as percentages. The higher the value is, the better the model is.
    #model_2_product = 'model2_P_cropEqual'+str(crop)+'.sav'
    
    #pickle.dump(model2, open(path+model_2_product, 'wb'))
    
    #print("'{}' : pickle.load(open('static/modelos/{}', 'rb')),".format(str(crop),(model_2_product)))
    # salva o modelo no disco
    
    return (model2)


# In[ ]:


trainCols = ['Area',  'Crop']
targetCols = ['Production']

x2=df2[trainCols].values
y2=df2[targetCols].values


# In[ ]:


modeloT = productionML(x2,y2,'')
## R^2 de 0.003 um numero bem baixo. Nosso R-squared é um coeficiente que representa o quão bem 
# nossos valores estão fitando os dados originais possuindo um intervalo de valores de 0 a 1 para expressar isso.
# Quanto mais proximo de 1. melhor nossa regressão linear é.
# Nesse caso, quase 0 de score, representa o pior dos cenários.


# In[ ]:


# Fazendo um modelo de regressão linear para cada crop individual
trainCols = ['Area']
targetCols = ['Production']
modelVet = {}
for i in range(len(a)):
    modelVet[a[i][0]] = productionML(a[i][1][trainCols],
                              a[i][1][targetCols],a[i][0])


# In[ ]:


model_2_product = 'model2_P_cropEqual_Final.sav'
pickle.dump(modelVet, open(path+model_2_product, 'wb'))


# Metricas:
# - MAE (Mean absolute error) represents the difference between the original and predicted values extracted by averaged the absolute difference over the data set.
# - MSE (Mean Squared Error) represents the difference between the original and predicted values extracted by squared the average difference over the data set.
# - RMSE (Root Mean Squared Error) is the error rate by the square root of MSE.
# 
