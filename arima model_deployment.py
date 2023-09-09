# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:21:21 2023

@author: hemamalini
"""

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from sqlalchemy import create_engine
from  pmdarima import auto_arima

df  = pd.read_csv(r'C:\Users\hemamalini\OneDrive\Documents\360\project\Data_cleaned.csv')
stepfit = auto_arima(df['QUANTITY'], trace = True, suppress_warning = True)
stepfit.summary()
    
# Data Partition
Train = df.iloc[:-30]
Test = df.iloc[-30:]

Test.to_csv('test_arima.csv')
import os
os.getcwd()

df = pd.read_csv('test_arima.csv', index_col = 0)

# ARIMA with AR = 0, MA = 1
model1 = ARIMA(Train['QUANTITY'], order = (0, 0, 1))
res1 = model1.fit()

print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = len(Train)+ len(Test)-1
forecast_test = res1.predict(start = start_index, end = end_index)
#forecast_test.index = df.index[start_index:end_index+1]
print(forecast_test)

forecast_test.plot(legend= True)
Test['QUANTITY'].plot(legend = True)

mean = Test['QUANTITY'].mean()
print(mean)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.QUANTITY, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.QUANTITY)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()

model2 = ARIMA(Train['QUANTITY'], order = (0, 0, 1))
res = model2.fit()

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)

print(forecast_best)

# checking both rmse of with and with out autoarima

print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)

# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model_new1.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model_new1.pickle")


