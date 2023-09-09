# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:32:15 2023

@author: hemamalini
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.formula.api as smf

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
a = pd.read_csv(r'C:\Users\hemamalini\OneDrive\Documents\360\project\Data_cleaned.csv')


df2=a[['SO Due Date','QUANTITY']]

df2 = df2.reset_index()

df2 = df2.drop(['index'], axis=1)

a["t"] = np.arange(1, 1272) # Linear Trend is captured
a["t_square"] = a["t"] * a["t"] # Quadratic trend or polynomial with '2' degrees trend is captured
a["log_QUANTITY"] = np.log(a["QUANTITY"])  # Exponential trend is captured
a.columns


Train=a.head(1200)
Test=a.tail(72)
Test

###### Linear Model ######

linear_model = smf.ols('QUANTITY ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['QUANTITY']) - np.array(pred_linear))**2))
print("linear model = ", rmse_linear)  

# 56.32862968469264

##### Exponential Model #####

Exp = smf.ols('log_QUANTITY ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['QUANTITY']) - np.array(np.exp(pred_Exp)))**2))
print("exponential model = ", rmse_Exp)
# 24.282627947234445

#### Quadratic Model ####

Quad = smf.ols('QUANTITY ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['QUANTITY']) - np.array(pred_Quad))**2))
print("Quadratic model = ", rmse_Quad)

#75.97196121361233
############## HOLT model ####################

def MAPE(pred,org): # MAPE = Mean absolute percentage error.
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)
hw_model = Holt(Train["QUANTITY"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
print("mape value ",MAPE(pred_hw, Test.QUANTITY) )

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["QUANTITY"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
print("mape value for Holts winter exponential smoothing with  additive seasonality and additive trend:",MAPE(pred_hwe_add_add, Test.QUANTITY) )

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["QUANTITY"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
print("mape value with exponential smoothing with multiplicative seasonality and additive trend:",MAPE(pred_hwe_mul_add, Test.QUANTITY) )


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(df2["QUANTITY"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_csv(r"C:/Users/hemamalini/OneDrive/Documents/360/project/Data_cleaned.csv")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
print("final model ",newdata_pred)


a['SO Due Date'] = pd.to_datetime(a['SO Due Date'])
a.set_index('SO Due Date', inplace=True)
a_monthly = a.resample('M').sum()


###### Model Based Approach ######## 

plt.figure(figsize=(10, 6))
plt.plot(a.index, a['QUANTITY'])
plt.xlabel('Date')
plt.ylabel('Pallet Demand')
plt.title('Historical Pallet Demand')
plt.show()

# Plot ACF
plt.figure(figsize=(10, 4))
plot_acf(a['QUANTITY'], lags=40)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
plt.figure(figsize=(10, 4))
plot_pacf(a['QUANTITY'], lags=40)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

def stationarity_test(series):
    # Augmented Dickey-Fuller (ADF) test
    adf_result = adfuller(series)
    print("ADF Test:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Critical Values:")
    for key, value in adf_result[4].items():
        print(f"  {key}: {value}")

    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    kpss_result = kpss(series)
    print("\nKPSS Test:")
    print(f"KPSS Statistic: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")
    print(f"Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"  {key}: {value}")

    # Interpret the results
    if adf_result[1] <= 0.05:
        print("\nThe time series is stationary (reject null hypothesis) based on the ADF test.")
    else:
        print("\nThe time series is non-stationary (fail to reject null hypothesis) based on the ADF test.")

    if kpss_result[1] <= 0.05:
        print("The time series is non-stationary (reject null hypothesis) based on the KPSS test.")
    else:
        print("The time series is stationary (fail to reject null hypothesis) based on the KPSS test.")
stationarity_test(a['QUANTITY'])
# First-order differencing
a['QUANTITY_DIFF'] = a['QUANTITY'].diff()

a['seasonal_DIFF'] = a['QUANTITY']-a['QUANTITY'].shift(12)

 #test dickey fuller test
print(adfuller(a['seasonal_DIFF'].dropna()))
#print(adfuller(a['QUANTITY']))
a['seasonal_DIFF'].plot()


#####AUTO REGRESSION MODEL#######
import pandas.plotting as pd_plotting
pd_plotting.autocorrelation_plot(a['QUANTITY'])

plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(a['seasonal_DIFF'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(a['seasonal_DIFF'].iloc[13:],lags=40,ax=ax2)

# For non-seasonal data
#p=1, d=1, q=0 or 1


# Assuming you have a pandas DataFrame called 'df' with a column named 'QUANTITY'
# Specify the order of the ARIMA model (p, d, q)
order = (0, 0, 1)


######ARIMA model#######
# Create and fit the ARIMA model
model = SARIMAX(a['QUANTITY'], order=order)
model_fit = model.fit()
Train = a.head(1250)
Test = a.tail(20)


# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = model_fit.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
#rmse_test = np.sqrt(mean_squared_error(Test.QUANTITY, forecast_test))
#print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
plt.plot(Test.QUANTITY)
plt.plot(forecast_test, color='red')
plt.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user


ar_model = pm.auto_arima(Train.QUANTITY, start_p=0, start_q=0,
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)

# Best Parameters ARIMA
# ARIMA with AR=3, I = 1, MA = 5
model = ARIMA(Train.QUANTITY, order = (0,0,1))
res = model.fit()
print(res.summary())


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
plt.plot(Test.QUANTITY)
plt.plot(forecast_test, color = 'red')
plt.show()

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



################# Data Drievn Model ###########################

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = a.head(1200)
Test = a.tail(72)



# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org): # MAPE = Mean absolute percentage error.
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

a.QUANTITY.plot() # time series plot 

####### Moving Average for the time series############
mv_pred = a["QUANTITY"].rolling(12).mean()
mv_pred.tail(12)
MAPE(mv_pred.tail(12), Test.QUANTITY) ## MAPE = 31.4512 

# Plot with Moving Averages
a.QUANTITY.plot(label = "org")
for i in range(2, 9, 2):
    a["QUANTITY"].rolling(i).mean().plot(label = str(i))
plt.legend(loc='upper right' )

from statsmodels.tsa.seasonal import seasonal_decompose
# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(a.QUANTITY, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()



#########Simple Exponential Smoothing########

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

data_array = np.asarray(a['QUANTITY'])
# Split the data into training and testing sets
Train = data_array[:-1200]
Test = data_array[-70:]   # Testing data, where 'n' is the number of test points you want to forecast


# Create and fit the Simple Exponential Smoothing model
model = SimpleExpSmoothing(Train)
model_fit = model.fit()

# Forecast the future values for 10 months
forecast = model_fit.forecast(10)

# Print the forecasted values
print(forecast)

plt.plot(a.index, a['QUANTITY'], label='Actual Data')
plt.plot(forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

model_fit.save("model_new2.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model_new2.pickle")


