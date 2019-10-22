#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import modules
import numpy as np # Numeric operations
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Plots
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings("ignore")

## Location to models in api.

path='WEB MachineLearning model deployed with Flask/static/modelos/'


# In[13]:


# Read the initial dataset
apy = pd.read_csv('apy.csv')
"""
Context
Historical data of Indian agricultural
production on various location acquired from the Indian government web page.
https://data.gov.in""";


# In[14]:


# A Look in the dataframe
apy.head()


# In[15]:


# Check Shape
apy.shape


# In[16]:


# Check Variable types
apy.dtypes


# In[17]:


# check for null values
apy.isnull().sum()


# In[18]:


#Years in apy dataset
#sorted((apy.Crop_Year.value_counts().index))


# In[19]:


gdp = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_247793.csv', skiprows=4)


# In[20]:


gdp.head(3)


# In[21]:


# Locate India
gdp.loc[gdp['Country Name'] == 'India']


# In[22]:


# Select only 1997-2015
start_year = gdp.columns.get_loc('1997')
end_year = gdp.columns.get_loc('2015')
# India is row 107
gdp_9715 = gdp.iloc[107, start_year:end_year+1]


# In[23]:


# Create a Dataframe for gdp
gdp_9715_idx = gdp_9715.index[:]
gdp_dict = {'Year': gdp_9715_idx, 'GDP': gdp_9715}
gdp_df = pd.DataFrame(gdp_dict)
gdp_df = gdp_df.reset_index(drop=True)
gdp_df.head()


# In[24]:


# Check if Crop_Year have all values from 1997 to 2015
apy_crop_year = apy['Crop_Year']
apy_crop_year.unique()


# In[25]:


# Check Datatypes
gdp_df.dtypes


# In[26]:


# Convert year to int
gdp_df['Year'] = gdp_df['Year'].astype(int)


# In[27]:


# Map all GDP values to the respective year
rename_dict = gdp_df.set_index('Year').to_dict()['GDP']
apy_crop_year = apy_crop_year.replace(rename_dict)
# Create the definitive GDP Dataframe
apy_crop_year_dict = {'GDP': apy_crop_year}
gdp_final = pd.DataFrame(apy_crop_year_dict)


# In[28]:


# See the data type
gdp_final['GDP'].dtype


# In[29]:


# Concatenate the DataFrames
india_crop_gdp_1997_2015 = pd.concat([apy, gdp_final], axis=1, sort=False)


# In[30]:


# A look into DataFrame
india_crop_gdp_1997_2015.head()


# In[31]:


# Save as .csv
india_crop_gdp_1997_2015.to_csv('india_crop_gdp_1997_2015.csv')


# In[32]:


india_crop_gdp_1997_2015= pd.read_csv('india_crop_gdp_1997_2015.csv')
india_crop_gdp_1997_2015.columns


# In[33]:


# Label Enconding
from sklearn import preprocessing
RelationItens ={}
for f in india_crop_gdp_1997_2015.columns:
    if f == 'Production':
        continue
    if india_crop_gdp_1997_2015[f].dtype =='object': 
        vet = []
        le = preprocessing.LabelEncoder()
        itens = (sorted(list(india_crop_gdp_1997_2015[f].value_counts().index)))
        le.fit(list(india_crop_gdp_1997_2015[f].values))
        itens_edited = le.transform(itens)
        vet.append([[a,b] for a,b in zip (itens,itens_edited)])
        india_crop_gdp_1997_2015[f] = le.transform(list(india_crop_gdp_1997_2015[f].values))
        RelationItens[f]=vet
#del RelationItens['Production']
RelationItens.keys()


# In[42]:


#for i in RelationItens['Season'][0]:
#    print ('        <option value="{}">{}</option>'.format(i[1],i[0]))


# # Now we can train the data!
# 
# ## For the first algorithm (Classification problem)

# In[43]:


# Select variables (Features)
X1 = india_crop_gdp_1997_2015[['Area', 'District_Name', 'Season']]
y1 = india_crop_gdp_1997_2015['Crop']


# # Model 1 - Classification (Decision Tree Classifier)
# 
# - Algoritmo 1:  Objetivo: descobrir a CROP (Cultivo). A partir de 3 dados aleatórios (AREA, STATE, DISTRICT) inseridos pelo usuário, seja calculado o resultado de CROP

# In[44]:


# Split Train and Test
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, 
                                                    test_size = 0.15,
                                                   random_state = 42)

model1= DecisionTreeClassifier(random_state=42)
model1.fit(x1_train, y1_train)


# In[45]:


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





# In[46]:


#------   Saving the model with pickle -----------------
# Define o nome do arquivo em disco que irá guardar o nosso modelo
filename = 'model_1_FindCrop.sav'
# salva o modelo no disco
pickle.dump(model1, open(path+filename, 'wb'))


# # Model 2 - Regression (Decision Tree Regressor)
# 
# - Algoritmo 2: Objetivo: descobrir a PRODUCTION (Produção agrícola de um cultivo). A partir de 3 dados aleatórios (AREA, CROP e GDP) inseridos pelo usuário, seja calculado o resultado de PRODUCTION

# In[47]:


# Select variables (Features)
india_crop_gdp_1997_2015_2 = india_crop_gdp_1997_2015[india_crop_gdp_1997_2015['Production'] != '=']

x2 = india_crop_gdp_1997_2015_2[['Area', 'Crop', 'GDP']]
y2 = india_crop_gdp_1997_2015_2['Production']


# In[48]:


# Split Train and Test
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, 
                                                    test_size = 0.30,
                                                   random_state = 42)


# In[49]:


# Train the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

model2 = DecisionTreeRegressor(random_state=42)
model2.fit(x2_train, y2_train)


# In[50]:


# Define o nome do arquivo em disco que irá guardar o nosso modelo
model_2_product = 'model2_P.sav'
# salva o modelo no disco
pickle.dump(model2, open(path+model_2_product, 'wb'))


# In[51]:


# Calculate the metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error


y2_pred = model2.predict(x2_test)
mae = mean_absolute_error(y2_test, y2_pred)
mse = mean_squared_error(y2_test, y2_pred)
r2 = r2_score(y2_test, y2_pred)
mad = median_absolute_error(y2_test, y2_pred)
print('MAE: ', mae)
print('MSE: ', mse)
print('r2: ', r2)
print('MAD: ', mad)


# Metricas:
# - MAE (Mean absolute error) represents the difference between the original and predicted values extracted by averaged the absolute difference over the data set.
# - MSE (Mean Squared Error) represents the difference between the original and predicted values extracted by squared the average difference over the data set.
# - RMSE (Root Mean Squared Error) is the error rate by the square root of MSE.
# - R-squared (Coefficient of determination) represents the coefficient of how well the values fit compared to the original values. The value from 0 to 1 interpreted as percentages. The higher the value is, the better the model is.

# In[ ]:




