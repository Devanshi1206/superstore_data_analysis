#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore') # To prevent kernel from showing any warning


# # **Importing Dataset**

# In[3]:


df=pd.read_csv('train.csv')
df.head()


# ## EDA

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(subset = ['Postal Code']) 


# In[8]:


df.describe()


# In[9]:


df['Product Name'].value_counts()


# In[10]:


df.duplicated().sum()


# In[11]:


df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')   
df['order_day'] = df['Order Date'].dt.day
df['ship_day'] = df['Ship Date'].dt.day
df['order_year'] = df['Order Date'].dt.year
df['ship_year'] = df['Ship Date'].dt.year
df['Fulfillment_Time'] = df['Ship Date'] - df['Order Date']
df['Fulfillment_Time'] = df['Fulfillment_Time'].dt.days
df['year_month']= df['Order Date'].apply(lambda x: x.strftime("%Y-%m"))


# # Data Visualization

# In[12]:


# Creating plot
plt.boxplot(df['Sales'])
 
# show plot
plt.show()


# In[13]:


ship_mode_counts = df['Ship Mode'].value_counts()

plt.figure(figsize=(8, 6))
ship_mode_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title('Distribution of Ship Modes')
plt.ylabel('')
plt.show()


# In[14]:


prod = df[['Category', 'Sub-Category', 'Product Name', 'Sales', 'order_year']]

fig = px.sunburst(
    prod, path=["Category", "Sub-Category"], values="Sales",
    color="Category",
    title="Sales by Category and Subcategory",
    width = 600,
    height = 600
)

fig.show()


# In[15]:


cust = df[['Order ID','Customer ID', 'Segment', 'Ship Mode', 'State', 'Sales', 'order_year', 'Order Date']]


# In[16]:


state_sales = cust.groupby(['State'])['Sales'].sum().sort_values(ascending=False).head(10)

fig = go.Figure(data=[go.Pie(labels=state_sales.index, values=state_sales.values)])
fig.update_traces(textposition='inside', textinfo='percent+label')


# 
# 1. We can clearly see that the Technology category is the most popular of all three categories and has achieved the highest sales over the years despite having only two subcategories.
# 2. In contrast, “Office Supplies” has the most subcategories but still has the lowest sales.
# Taking into account our new Fulfillment_Time feature, we can see that 2018 was the best year as this year saw the fastest shipments for most products.
# 3. “Tech” and “Furniture” also had very good delivery speeds.
# 4. We can see that most customers prefer standard shipping over first class or same day delivery. This means that we can focus on targeted promotions and monitor customer behavior to analyze the reasons for choosing this method, such as: B. Cost sensitivity, product urgency, regional distribution.

# # Trends 

# In[17]:


numeric_columns = df.select_dtypes(include=['number']).columns
a = df.groupby('order_year')[numeric_columns].sum()
plt.figure(figsize=(12, 4))
sns.pointplot(x= a.index, y='Sales', data=a)

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title("Total Sales per Year")

plt.show()


# In[18]:


a


# In[19]:


numeric_columns = df.select_dtypes(include=['number']).columns
a= pd.DataFrame(df.groupby('year_month')[numeric_columns].sum())['Sales']

plt.figure(figsize=(12, 4))
a.plot(kind='line')

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title("Total Sales per Year")
plt.show()


# In[20]:


a.head()


# # Model Building

# In[21]:


# importing libraries
# from xgboost.xgbclassifier import XGBClassifier
from xgboost import XGBRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error


# In[22]:


df= df.drop(["Row ID","Order ID","Customer ID","Product ID"],axis=1)


# In[23]:


from sklearn.model_selection import train_test_split

x = df.drop('Sales', axis=1)
y = df['Sales']


# ## XGBoost

# In[24]:


# Select only numeric columns for the groupby operation
numeric_columns = df.select_dtypes(include=['number']).columns
xgb_sales = pd.DataFrame(df.groupby(by=['Order Date'])[numeric_columns].sum())

x = xgb_sales.drop('Sales', axis=1)
y = xgb_sales['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

model = XGBRegressor(learning_rate=0.03, max_depth=1,)
model.fit(x_train, y_train)

preds = model.predict(x_test)
rmse_xgb = sqrt(mean_squared_error(y_test, preds))
model_score = model.score(x_test, y_test)

print("Root Mean Squared Error for XGBoost:", rmse_xgb)
print("Model Score:", model_score)


# ## Random Forest Regressor

# In[25]:


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Assuming x and y are defined from your previous preprocessing steps

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(x_train, y_train)

# Predict on test data
preds = rf_model.predict(x_test)

# Calculate evaluation metrics
rmse_rf = sqrt(mean_squared_error(y_test, preds))
mae_rf = mean_absolute_error(y_test, preds)
r2_rf = r2_score(y_test, preds)

# Print evaluation metrics
print("Root Mean Squared Error for Random Forest:", rmse_rf)
print("Mean Absolute Error for Random Forest:", mae_rf)
print("R-squared for Random Forest:", r2_rf)


# ## Comparing Result

# In[26]:


cust['latest_order_year'] = cust.groupby('Customer ID')['order_year'].transform('max')

cust['churn'] = (cust['order_year'] != cust['latest_order_year'])
churn_rate = cust[cust['churn'] == True].shape[0] / len(cust)
pct = churn_rate * 100

print(f"Churn Rate: {pct:.2f} % ")


# In[27]:


total_revenue = cust['Sales'].sum()
total_customers = len(cust['Sales'].unique())
RPU = total_revenue / total_customers
print(RPU)


# In[28]:


result = pd.DataFrame([[rmse_xgb], [rmse_rf]], columns=['RMSE'], index=['XGBRegressor','RandomForest'])
result

