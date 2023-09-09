# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:42:39 2023

@author: hemamalini
"""

import pandas as pd               # It is for importing data & performing sevaral operation about the dataframe
import numpy as np                # It is for to access numbers & arrays
import seaborn as sns             # data advance visualisation
import matplotlib.pyplot as plt   # for plotting various plot/graphs like bar plot, histogram, boxplot.
from scipy import stats           # Performing the stats operation

df = pd.read_csv(r'C:\Users\hemamalini\Downloads\Pallets data (1)\Palletdata.csv')
print(df)

df["CustomerVendor Code"].value_counts()# drop that col 
df["CustomerVendor Name"].value_counts() # drop that col 
df["LOB"].value_counts() # drop that col 
df["Region"].value_counts() # drop that col 
df["City"].value_counts() # drop that col 
df["STATE"].value_counts() # drop that col 
df["FromWhsCode"].value_counts()
df["FromWhsName"].value_counts()
df["TowhsCode"].value_counts() # drop that col 
df["TOWhsName"].value_counts() # drop that col 
df["ModelTYPE"].value_counts() 
df["TransferType"].value_counts() # drop that col 
df["U_Frt"].value_counts()
df["U_ActShipType"].value_counts()
df["PRODUCTCATEGORY"].value_counts()
df["ItemCode"].value_counts()
df["Description"].value_counts()
df["QUANTITY"].value_counts()
df["UNIT"].value_counts() # drop that col 
df["RATE"].value_counts()
df["SOID"].value_counts()
df["U_DocStatus"].value_counts() # drop that col 
df["BPCATEGORY"].value_counts() # drop that col 
df["DocumentType"].value_counts() # drop that col 
df["TRANSPORTERNAME"].value_counts()
df["U_GRNNO"].value_counts()
df["LoadingUnloading"].value_counts()
df["Detention"].value_counts() # drop that col 
df["KITITEM"].value_counts()  # drop that col 
df["U_AssetClass"].value_counts()
df["CustomerType"].value_counts() # drop that col 
df["U_TRINPD"].value_counts()
df["U_SOTYPE"].value_counts()

df = df.drop(['CustomerVendor Code', 'CustomerVendor Name', 'LOB', 'Region', 'BPTYPE', 'City', 'STATE', 'TowhsCode', 'TOWhsName', 'TransferType', 'UNIT', 'U_DocStatus', 'U_SOTYPE', 'BPCATEGORY', 'DocumentType', 'Detention', 'KITITEM', 'CustomerType', 'U_TRINPD'], axis=1)

df.dtypes   


df.info()
#df.to_excel('C:/Users/hemamalini/Downloads/Pallets data (1)/new_data.xlsx', index=False)
# check for count of NA'sin each column
print(df.isna().sum())

from sklearn.impute import SimpleImputer
# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["U_AssetClass"] = pd.DataFrame(mode_imputer.fit_transform(df[["U_AssetClass"]]))
df["U_AssetClass"].isna().sum()

df.isnull().sum()

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["U_Frt"] = pd.DataFrame(mean_imputer.fit_transform(df[["U_Frt"]]))
df["U_Frt"].isna().sum()

df["U_GRNNO"] = pd.DataFrame(mean_imputer.fit_transform(df[["U_GRNNO"]]))
df["U_GRNNO"].isna().sum()

# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["LoadingUnloading"] = pd.DataFrame(median_imputer.fit_transform(df[["LoadingUnloading"]]))
df["LoadingUnloading"].isna().sum()  

df.isna().sum()

# Find Duplicates
duplicate = df.duplicated()
duplicate
sum(duplicate)
    
# Removing duplicates 

df1 = df.drop_duplicates()

df1.U_Frt.mean()
df1.QUANTITY.mode()
df1.RATE.mean()
df1.U_GRNNO.mean()
df1.LoadingUnloading.mean()

# Find Median
df1.U_Frt.median()
df1.RATE.median()

df1.U_GRNNO.median()
df1.LoadingUnloading.median()


pd.options.mode.chained_assignment = None
df1['QUANTITY'] = df1['QUANTITY'].str.replace(',', '')  # Remove commas from the string values
df1['QUANTITY'] = pd.to_numeric(df1['QUANTITY'], errors='coerce')  # Convert to numeric type

# Select only the 'QUANTITY' column and fill missing values with the mean
df1[['QUANTITY']] = df1[['QUANTITY']].fillna(df1[['QUANTITY']].mean())
#df1.loc[:, 'QUANTITY'] = pd.to_numeric(df1['QUANTITY'], errors='coerce')

# We can find the mode by two methods 1. by in built mode function & 2. by importing stats library
df1.FromWhsCode.mode()
df1.ModelTYPE.mode()
df1.U_ActShipType.mode()
df1.PRODUCTCATEGORY.mode()
df1.ItemCode.mode()
df1.Description.mode()
df1.SOID.mode()
df1.TRANSPORTERNAME.mode()
df1.U_AssetClass.mode()

# pip install numpy
from scipy import stats
# Find mode



# Measures of Dispersion / Second moment business decision
# Find Variance
df1.U_Frt.var() # variance
df1.QUANTITY.var()
df1.RATE.var()
df1.U_GRNNO.var()
df1.LoadingUnloading.var()

# Find standard deviation
df1.U_Frt.std() # standard deviation
df1.QUANTITY.std()
df1.RATE.std()
df1.U_GRNNO.std()
df1.LoadingUnloading.std()


# Find RANGE
range1 = max(df1.U_Frt) - min(df1.U_Frt) # range
range1
range2 = max(df1.QUANTITY) - min(df1.QUANTITY) # range
range2
range3 = max(df1.RATE) - min(df1.RATE) # range
range3
range4 = max(df1.U_GRNNO) - min(df1.U_GRNNO) # range
range4
range5 = max(df1.LoadingUnloading) - min(df1.LoadingUnloading) # range
range5
# Third moment business decision
# Find Skewness
df1.U_Frt.skew()
df1.QUANTITY.skew()
df1.RATE.skew()
df1.U_GRNNO.skew()
df1.LoadingUnloading.skew()


# Fourth moment business decision
# Find Kurtosis
df1.U_Frt.kurt()
df1.QUANTITY.kurt()
df1.RATE.kurt()
df1.U_GRNNO.kurt()
df1.LoadingUnloading.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

# Plotting BAR PLot
df1.shape
plt.bar(height = df1.U_Frt, x = np.arange(1, 1023, 1)) # initializing the parameter
plt.bar(height = df1.QUANTITY, x = np.arange(1, 1023, 1)) # initializing the parameter
plt.bar(height = df1.RATE, x = np.arange(1, 1023, 1)) # initializing the parameter
plt.bar(height = df1.U_GRNNO, x = np.arange(1, 1023, 1)) # initializing the parameter
plt.bar(height = df1.LoadingUnloading, x = np.arange(1, 1023, 1)) # initializing the parameter

# Plotting Hisogram PLot
plt.hist(df1.U_Frt) #histogram
plt.hist(df1.QUANTITY, color='red')
plt.hist(df1.RATE, color='green') 
plt.hist(df1.U_GRNNO, color='orange')
plt.hist(df1.LoadingUnloading, color='yellow') 
help(plt.hist)




