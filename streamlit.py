import streamlit as st
import pandas as pd
#from statsmodels.tsa.arima.model import ARIMA
import pickle
import datetime
import altair as alt
import yfinance as yf
# import numpy as np
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model_new1.pickle")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def main():
    

    st.title("Prediction of Pallets Profit")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile,  index_col=0)
        except:
                try:
                    data = pd.read_excel(uploadedFile,  index_col=0)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    
    if st.button("Predict"):
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        
        
        ###############################################
        st.subheader(":red[Forecast for Test data]", anchor=None)
         
        forecast_test = pd.DataFrame(model.predict(start = data.index[0], end = data.index[-1]))
        results = pd.concat([data,forecast_test], axis=1)
        results.to_sql('forecast_results', con = engine, if_exists = 'replace', index = False, chunksize = 1000)
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.line_chart(results)        
        ###############################################
        st.text("")
        st.subheader(":red[plot forecasts against actual outcomes]", anchor=None)
        #plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(data.QUANTITY)
        ax.plot(forecast_test, color = 'red')
        st.pyplot(fig)
        
        ###############################################
        st.text("")
        st.subheader(":red[Forecast for the nest 12 months]", anchor=None)
        
        forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))
        st.table(forecast)
# Load the ARIMA model and other necessary data
#model = pickle.load(open(r'C:\Users\hemamalini\OneDrive\Documents\360\project\project_deployment\model_new.pickle', encoding='latin1') ) # Replace 'arima_model.pkl' with the actual path to your ARIMA model file

with open(r'C:\Users\hemamalini\OneDrive\Documents\360\project\project_deployment\model_new1.pickle','rb') as file:
    
    model = pickle.load(file)
data = pd.read_csv(r'C:\Users\hemamalini\OneDrive\Documents\360\project\Data_cleaned.csv') # Replace 'data.csv' with the actual path to your data file

# Streamlit app
st.title('Profit Calculator of the Pallets')

# Sidebar inputs
#data['SO Creation Date'] = pd.to_datetime(data['SO Creation Date'])
data['SO Due Date'] = pd.to_datetime(data['SO Due Date'])


#start_date = st.date_input('Start Date')
#end_date = st.date_input('End Date')

start_date = pd.to_datetime(data['SO Creation Date'])
end_date = pd.to_datetime(data['SO Due Date'])

# Calculate profit
filtered_data = data[(data['SO Due Date'] >= start_date) & (data['SO Due Date'] <= end_date)]
forecast = model.predict(start=len(data), end=len(data) + len(filtered_data) - 1)  # Adjust this line based on your ARIMA model's predict method
filtered_data['Forecast'] = forecast
filtered_data['Profit'] = filtered_data['QUANTITY'] * filtered_data['RATE']  # Adjust this line based on your data columns

#tickers = ('wooden Pallet', 'Belts & Wedges')
#dropdown = st.multiselect('Pick your Asset class', tickers)

#start = st.date_input('start', value = pd.to_datetime('2019-01-01'))
#end = st.date_input('start', value = pd.to_datetime('today'))

filtered_data['SO Due Date'] = pd.to_datetime(filtered_data['SO Due Date'])

# Group the data by year and calculate the sum of profits
profit_year_data = filtered_data.groupby(filtered_data['SO Due Date'].dt.year)['Profit'].sum().reset_index()

st.bar_chart(profit_year_data, x='SO Due Date', y='Profit')

chart = alt.Chart(filtered_data).mark_line().encode(
    x='SO Due Date:T',
    y='Profit:Q',
    tooltip=['SO Due Date', 'Profit']
).interactive()

# Set the y-axis format to display dates
chart = chart.configure_axisY(
    labelExpr="toDate(datum.value)",
    labelPadding=10
)

# Render the chart using Streamlit
st.altair_chart(chart, use_container_width=True)

                           
if __name__=='__main__':
    main()