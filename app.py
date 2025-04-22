import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Stock Price Predicton App')
st.sidebar.header('User Selection')
stock_ticker = st.sidebar.text_input('Enter stock Ticker(e.g,RELIANCE BO):','RELIANCE.BO')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date',pd.to_datetime('2025-04-01'))


stock_data = yf.download(stock_ticker, start=start_date,end=end_date)
stock_data.reset_index(inplace=True)
stock_data['Days'] = (stock_data.index - stock_data.index.min())

stock_data = stock_data[['Days','Date','Close']]


x = stock_data.drop(['Date','Close'], axis = 'columns')
y = stock_data['Close']
model = LinearRegression()
model.fit(x,y)

st.sidebar.subheader('Future Prediction Input')
select_date = st.sidebar.date_input('Select Date',pd.to_datetime('2025-04-01'))

days_diff = (pd.to_datetime(select_date) - stock_data['Date'].min()).days

y_pred = model.predict([[days_diff]])
st.write(f'Predicted Close Price on {select_date} is: ₹{y_pred[0]}')

hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
