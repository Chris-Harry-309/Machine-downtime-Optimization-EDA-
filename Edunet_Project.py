#!/usr/bin/env python
# coding: utf-8

# ### Exploratory data analysis using python on "Machine downtime optimization"

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


## To Read the CSV file 
df = pd.read_csv(r"C:\Users\CHRIS\Desktop\Project\Machine Downtime.csv", index_col = "Date")
df


# In[3]:


print(df.head())


# In[4]:


print(df.tail())


# In[5]:


print(df.describe())


# In[6]:


df.shape


# In[7]:


df.info


# In[9]:


df.columns


# In[10]:


## To Display the data types of each column in the DataFrame
df.dtypes


# ### Data Cleaning

# In[5]:


# for Checking all missing values in all columns
missing_val = df.isnull().sum()
print("Missing values in each of the columns:")
print(missing_val)


# In[15]:


#Handling the missing values
for column in df.columns:
    if df[column].dtype == 'object':
# For categorical columns we are replacing missing values with the mode value of that column
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
    else:
# For numerical columns we are replacing missing values with the mean value of that column
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
print("\nDataFrame after handling missing values:")
print(df)


# In[6]:


#This code is to check we have handled all the missing values for both categorical and numerical variables in our dataset
missing_val = df.isnull().sum()
print("Missing values in each of the columns:")
print(missing_val)


# In[17]:


# Convert categorical variables to numerical representations for data compatability
categorical_columns = ['Machine_ID', 'Assembly_Line_No', 'Downtime']
for column in categorical_columns:
    df[column] = pd.factorize(df[column])[0]
print("\nDataFrame after converting categorical variables to numerical representations:")
print(df)


# ## Computing Business moments in cleaned dataset

# In[34]:


import pandas as pd
from scipy.stats import kurtosis, skew
df = pd.read_csv('preprocessed_data.csv')
df
numeric_cols = df.select_dtypes(include=['number'])
means = numeric_cols.mean()
medians = numeric_cols.median()
modes = numeric_cols.mode().iloc[0]  
kurtoses = numeric_cols.apply(lambda x: kurtosis(x, nan_policy='omit'))
skewnesses = numeric_cols.apply(lambda x: skew(x, nan_policy='omit'))
ranges = numeric_cols.max() - numeric_cols.min()
for column in numeric_cols.columns:
    print("Variable:", column)
    print("Mean:", means[column])
    print("Median:", medians[column])
    print("Mode:", modes[column])
    print("Kurtosis:", kurtoses[column] if not pd.isna(kurtoses[column]) else "Not Available")
    print("Skewness:", skewnesses[column] if not pd.isna(skewnesses[column]) else "Not Available")
    print("Range:", ranges[column])
    print("\n")


# In[18]:


#checking for dupliocate value if it exist.
df.duplicated().sum()


# In[74]:


df = pd.read_csv(r"C:\Users\CHRIS\Desktop\Project\Machine Downtime.csv", index_col="Date")
# Plotting outliers of each variable using box plots
plt.figure(figsize=(18, 12))
df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(18, 12), sharex=False, sharey=False)
plt.xticks(rotation=60)
plt.suptitle('Box Plots - Outlier Detection for Each Variable')
plt.tight_layout(rect=[0, 0, 1, 4])
plt.show()


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\CHRIS\Desktop\Project\Machine Downtime.csv", index_col="Date")
df
numerical_vars = ['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)', 
                  'Coolant_Temperature', 'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)',
                  'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)', 'Voltage(volts)',
                  'Torque(Nm)', 'Cutting(kN)']
#scatter plot
for var in numerical_vars:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=var, y=var)
    plt.title(f'Scatter Plot for {var}')
    plt.xlabel(var)
    plt.ylabel(var)
    plt.show()

# Box plots for numerical variables to detect outliers
for var in numerical_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[var], orient='h', palette='Set2')
    plt.title(f'Box Plot for {var}')
    plt.xlabel(var)
    plt.show()


# In[37]:


## computing the correlatrion matrix for our dataset
numeric_columns = df.select_dtypes(include='number')
column_corr = numeric_columns.corr()
print("Correlation of Columns:")
print(column_corr)


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(column_corr, cmap='cividis_r', xticklabels=column_corr.columns, yticklabels=column_corr.columns)
plt.title('Correlation Heatmap')
plt.show()


# In[49]:


#Ratio of downtime in dataset
df = pd.read_csv(r"C:\Users\CHRIS\Desktop\Project\Machine Downtime.csv", index_col="Date")
failure = df['Downtime'].value_counts()
plt.figure(figsize=(4, 4))
plt.pie(failure, labels=failure.index, autopct='%1.1f%%', startangle=90)
plt.title('Machine downtime Occurrences')
plt.show()


# In[52]:


machinedowntime_counts = df.groupby('Assembly_Line_No')['Downtime'].count()


# In[53]:


print(machinedowntime_counts)


# In[54]:


# pie chart showing downtime occurences by assembly line no.
plt.figure(figsize=(6, 6))
plt.pie(machinedowntime_counts, labels=machinedowntime_counts.index, autopct='%1.1f%%')
plt.title('Downtime by Assembly Line Number')
plt.show()


# In[48]:


columns = ['Hydraulic_Pressure(bar)','Coolant_Pressure(bar)','Air_System_Pressure(bar)', 
           'Coolant_Temperature', 'Hydraulic_Oil_Temperature(°C)', 'Spindle_Bearing_Temperature(°C)', 
           'Spindle_Vibration(µm)', 'Tool_Vibration(µm)', 'Spindle_Speed(RPM)', 'Voltage(volts)', 
           'Torque(Nm)', 'Cutting(kN)']


plt.figure(figsize=(10,30))
for i, col in enumerate(columns):
    plt.subplot(len(columns), 1, i + 1)
    sns.scatterplot(data=df, x=col, y='Downtime')
    plt.xlabel(col)
    plt.ylabel('Downtime')
    plt.title(f'{col} vs Downtime')


# In[ ]:




