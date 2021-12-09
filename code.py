#!/usr/bin/env python
# coding: utf-8

# # Starbucks Stores Analysis

# In[1]:


# Housekeeping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math


# ## Datasets

# Data Constraints:
# - Both Starbucks and US datasets published in 2017.
# - Starbucks store locations limited to US country. 
# - Starbucks store limited to Starbucks brand (no Teavana)
# - Exclude Puerto Rico from US datasets

# In[2]:


starbucks = pd.read_csv('data/directory.csv')
starbucks = starbucks.query("Brand == 'Starbucks'").query("Country == 'US'")
starbucks = starbucks.drop(columns=["Brand", "Store Name", "Ownership Type", "Street Address","Phone Number","Timezone", "Postcode", "Country"])
starbucks = starbucks.rename(columns={'State/Province' : 'State'})


# In[3]:


cities = pd.read_csv('data/uscities.csv')
cities = cities[["city", "state_id", "state_name", "county_name"]]


# In[4]:


demographic = pd.read_csv('data/demo.csv', encoding='cp1252')
demographic = demographic[demographic['State'] != 'Puerto Rico']
demographic["County"] = demographic["County"].apply(lambda x: ' '.join(x.split()[0:-1]))


# ### Data Clean Up

# In[5]:


mapping = pd.merge(starbucks, cities, left_on=["City", "State"], right_on=["city", "state_id"])
mapping = mapping.drop(columns=["state_id", "city", "State"])
mapping = mapping.rename(columns={"state_name":"State", "county_name":"County"})
mapping


# In[6]:


storecount = mapping.groupby(['County', 'State'])['Store Number'].count().to_frame().reset_index()
storecount = storecount.rename(columns={"Store Number":"Count"})
storecount


# In[7]:


df = storecount.merge(demographic, how='right', left_on=['County', 'State'], right_on=['County', 'State']).drop(columns=["Unnamed: 0", "CountyId", "VotingAgeCitizen"])
df['Count'] = df['Count'].fillna(0)
df['Men'] = (df['Men']/df['TotalPop'])*100
df['Women'] = (df['Women']/df['TotalPop'])*100
df['Employed'] = (df['Employed']/df['TotalPop'])*100


# ### Correlation

# In[8]:


var = ['Count', 'TotalPop', 'Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific', 'Income', 'IncomeErr',
       'IncomePerCap', 'IncomePerCapErr', 'Poverty', 'ChildPoverty',
       'Professional', 'Service', 'Office', 'Construction', 'Production',
       'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome',
       'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 'SelfEmployed',
       'FamilyWork', 'Unemployment']
corr = df[var].corr().drop('Count')[['Count']]
best_corr = corr[abs(corr["Count"])>.19]
best_corr


# ### Preprocessing

# In[9]:


feature_names = ['TotalPop', 'White', 'Asian', 'IncomePerCap', 'Professional', 'Construction', 'Production', 'Transit']
features = df[feature_names].fillna(0)
features = features.apply(lambda x: stats.zscore(x))
target = df[["Count"]]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)


# In[10]:


array_for_visualization = []

print('Cross Validation:')
for i in range(1,5):
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x_train, y_train, test_size=0.2)
    poly_model = make_pipeline(PolynomialFeatures(i), LinearRegression())
    d = pd.DataFrame(cross_validate(poly_model, x_train_temp, y_train_temp, scoring=('r2', 'neg_mean_squared_error')))
    print(d.mean())
    array_for_visualization.append(d.mean()['test_r2'])
    


# ### Linear Regression

# In[11]:

model = LinearRegression()
model.fit(x_train, y_train)

map_ = np.vectorize(lambda x: 0 if x < 0 else math.floor(x))
y_pred = np.array([map_(y) for y in model.predict(x_test)])

print(f'Accuracy (R Squared): {r2_score(y_test, y_pred)}')


# In[12]:


# from tensorflow.keras import models, layers

# model = models.Sequential()
# model.add(layers.Dense(128, activation='relu', input_shape=(len(feature_names),)))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(1, activation='linear'))
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs = 100, batch_size=32)
# results = model.evaluate(x_test, y_test)
# results


# ## Data Visualization

# In[13]:


def normalize(df, columns):
    result = df[columns]
    result = (result - result.mean())/result.std()
    return result

data = df[["TotalPop","White","Asian", "Income","IncomePerCap","Professional","Construction","Transit"]]
columns = ["TotalPop","White","Asian","Income","IncomePerCap","Professional","Construction","Transit"]
data = normalize(data, columns)
data = data.join(df['Count'])

# Feature = TotalPop
data.plot.scatter(x="TotalPop", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Total Population")
plt.title("Number of Starbucks by Feature")

# Feature = White Population
data.plot.scatter(x="White", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Proportion of Population that is White")
plt.title("Number of Starbucks by Feature")

# Feature = Asian Population
data.plot.scatter(x="Asian", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Proportion of Population that is Asian")
plt.title("Number of Starbucks by Feature")

# Feature = Income
data.plot.scatter(x="Income", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Income")
plt.title("Number of Starbucks by Feature")

# Feature = Income Per Capita
data.plot.scatter(x="IncomePerCap", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Income Per Capita")
plt.title("Number of Starbucks by Feature")

# Feature = Professional Population
data.plot.scatter(x="TotalPop", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Proportion of Population that are Professionals")
plt.title("Number of Starbucks by Feature")

# Feature = Construction Population
data.plot.scatter(x="Construction", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Proportion of Population that work in Construction")
plt.title("Number of Starbucks by Feature")

# Feature = White Population
data.plot.scatter(x="Transit", y="Count",s=1)
plt.ylabel("# of Starbucks")
plt.xlabel("Proportion of Population that work in Transit")
plt.title("Number of Starbucks by Feature")


# In[16]:


array_for_visualization
plot_data = pd.DataFrame(array_for_visualization).rename(columns = {0: 'r2'}).reset_index()
plot_data['index'] = [1, 2, 3, 4]
columns = ['index', 'r2']
plot_data.plot.scatter(x="index", y="r2",s=15)
plt.ylabel("R-Squared Value")
plt.xlabel("Degree Polynomial")
plt.title("Accuracy by Degree Polynomial")

