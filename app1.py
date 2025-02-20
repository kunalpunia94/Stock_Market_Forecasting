import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the title of the Streamlit app
st.title('Stock Forecast App')

# Add a brief description of the Prophet library and its advantages and disadvantages
st.markdown("""
Prophet is an open-source forecasting library developed by Facebook. It is designed to forecast time series data with strong seasonal patterns and several cycles of historical data.

Prophet has a number of advantages over other forecasting libraries, including:

* It is easy to use and requires no coding experience.
* It is accurate and performs well on a variety of time series data.
* It is fast and can generate forecasts in seconds.
* It is interpretable and provides insights into the factors that are contributing to the forecast.
""")

# Add a link to the Prophet documentation
st.markdown("""
To learn more about the Prophet library, please visit the [Prophet documentation](https://facebook.github.io/prophet/).
""")


def plot_forecast_components(m, forecast, stock_name):
    fig = m.plot_components(forecast)
    st.write(f'Forecast components for {stock_name}')
    st.write(fig)


# Define a function to download the forecast data as CSV files
def download_forecast_csv(forecast):
    forecast.to_csv('forecast.csv', index=False)

#-----------------------------------------------------------new_--------------------change
def clean_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data


# # Define a list of stock tickers to choose from
# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')

# # Create selection boxes for the user to choose datasets for prediction and comparison
# selected_stock = st.selectbox('Select dataset for prediction', stocks)
# selected_stock2 = st.selectbox('Select the second dataset for comparison', stocks)

stocks = ['GOOG', 'AAPL', 'MSFT', 'GME', 'Custom']

# # User selects from predefined stocks or chooses "Custom"
# selected_stock = st.selectbox('Select dataset for prediction', stocks)
# selected_stock2 = st.selectbox('Select the second dataset for comparison', stocks)

# # If "Custom" is selected, take user input for a stock symbol
# if selected_stock == "Custom":
#     selected_stock = st.text_input("Enter a stock symbol (e.g., TSLA, AMZN):").upper()

# if selected_stock2 == "Custom":
#     selected_stock2 = st.text_input("Enter second stock symbol:", key="custom_stock2").upper()



# User selects from predefined stocks or chooses "Custom"
selected_stock = st.selectbox('Select dataset for prediction', stocks)
selected_stock2 = st.selectbox('Select the second dataset for comparison', stocks)

# Initialize variables for custom stock input
custom_stock_1 = ""
custom_stock_2 = ""

# If "Custom" is selected, prompt for user input
if selected_stock == "Custom":
    custom_stock_1 = st.text_input("Enter a stock symbol (e.g., TSLA, AMZN):").upper()

if selected_stock2 == "Custom":
    custom_stock_2 = st.text_input("Enter second stock symbol:", key="custom_stock2").upper()

# Ensure valid stock selection
if selected_stock == "Custom" and not custom_stock_1:
    st.warning("Please enter a valid stock symbol for prediction.")
    st.stop()

if selected_stock2 == "Custom" and not custom_stock_2:
    st.warning("Please enter a valid second stock symbol for comparison.")
    st.stop()

# Assign the valid custom stock symbol to selected_stock
if selected_stock == "Custom":
    selected_stock = custom_stock_1

if selected_stock2 == "Custom":
    selected_stock2 = custom_stock_2






# Define the number of years for prediction and calculate the prediction period
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Define a function to load historical stock price data for a given ticker and cache the data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data = clean_data(data)  #-----------------------------------------change
    data.reset_index(inplace=True)
    return data

# Load data for the first selected stock
data_load_state = st.text('Loading data for the first selected stock...')
data = load_data(selected_stock)
data_load_state.text('Loading data for the first selected stock... done!')

# Train the Prophet model for the first selected stock
df_train = data[['Date', 'Close']].copy() #----------------------------------------change
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = Prophet()
m.fit(df_train)

# Generate the forecast for the first selected stock
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display the forecast data for the first selected stock
st.subheader(f'Forecast data for {selected_stock}')
st.write(forecast.tail())

# Plot the forecast for the first selected stock
st.write(f'Forecast plot for {n_years} years ({selected_stock})')
fig = plot_plotly(m, forecast)
st.plotly_chart(fig)

# Load data for the second selected stock
data_load_state2 = st.text('Loading data for the second selected stock...')
data2 = load_data(selected_stock2)
data_load_state2.text('Loading data for the second selected stock... done!')

# Train the Prophet model for the second selected stock
df_train2 = data2[['Date', 'Close']].copy() #----------------------------------------change
df_train2 = df_train2.rename(columns={"Date": "ds", "Close": "y"})

m2 = Prophet()
m2.fit(df_train2)

# Generate the forecast for the second selected stock
future2 = m2.make_future_dataframe(periods=period)
forecast2 = m2.predict(future2)

# Display the forecast data for the second selected stock
st.subheader(f'Forecast data for {selected_stock2}')
st.write(forecast2.tail())

# Plot the forecast for the second selected stock
st.write(f'Forecast plot for {n_years} years ({selected_stock2})')
fig2 = plot_plotly(m2, forecast2)
st.plotly_chart(fig2)


# Compare forecasts for the two selected stocks
st.subheader('Compare forecasts')

st.write('You can compare the forecasts for the two selected stocks by plotting them on the same chart. To do this, simply select the "Compare forecasts" checkbox below.')

# Create a checkbox for comparing forecasts
if st.checkbox('Compare forecasts'):
    # Create a new figure
    fig = go.Figure()

    # Add the forecast line plots for the two stocks
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=selected_stock))
    fig.add_trace(go.Scatter(x=forecast2['ds'], y=forecast2['yhat'], name=selected_stock2))

    # Update the figure layout
    fig.layout.update(title_text='Forecast comparison', xaxis_title='Date', yaxis_title='Closing price')

    # Plot the figure
    st.plotly_chart(fig)

# Plot the forecast components for the first selected stock
st.subheader(f'Forecast components for {selected_stock}')
plot_forecast_components(m, forecast, selected_stock)


# Plot the forecast components for the second selected stock
st.subheader(f'Forecast components for {selected_stock2}')
plot_forecast_components(m2, forecast2, selected_stock2)

# Show a button to download the forecast data as CSV files
st.button('Download forecast CSV', download_forecast_csv, args=(forecast, forecast2))




#finding the accuracy of the model-------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Remove future dates (keep only actual data for evaluation)
forecast_actual = forecast.merge(df_train, on="ds", how="inner")  # Join on actual dates

# Compute Error Metrics
mae = mean_absolute_error(forecast_actual['y'], forecast_actual['yhat'])
mse = mean_squared_error(forecast_actual['y'], forecast_actual['yhat'])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((forecast_actual['y'] - forecast_actual['yhat']) / forecast_actual['y'])) * 100

# Print the results
st.subheader("Model Accuracy Metrics:")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")




#------------------------------------change-----------------------
fig = go.Figure()

# Actual values
fig.add_trace(go.Scatter(x=forecast_actual['ds'], y=forecast_actual['y'],
                         mode='lines', name='Actual', line=dict(color='blue')))

# Predicted values
fig.add_trace(go.Scatter(x=forecast_actual['ds'], y=forecast_actual['yhat'],
                         mode='lines', name='Predicted', line=dict(color='red')))

fig.layout.update(title_text="Actual vs Predicted Stock Prices", xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig)




#-----------------perform cross validation----------------
from prophet.diagnostics import cross_validation, performance_metrics

# Perform Cross-Validation (Sliding Window Approach)
st.text("Performing cross-validation... This may take some time.")
df_cv = cross_validation(m, initial='730 days', period='365 days', horizon='180 days')
st.text("Cross-validation completed!")
st.text(df_cv.head())

# Compute Performance Metrics
df_p = performance_metrics(df_cv)

# Show the results
st.subheader("Cross-Validation Results:")
st.write(df_p[['horizon', 'mse', 'rmse', 'mae', 'mape']])

from prophet.plot import plot_cross_validation_metric

# Plot RMSE over different forecast horizons
fig = plot_cross_validation_metric(df_cv, metric='rmse')
st.subheader("Cross-Validation RMSE over Time")
st.pyplot(fig)
