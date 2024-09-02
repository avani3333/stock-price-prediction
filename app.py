import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set page config
st.set_page_config(page_title="Stock Price Prediction App", layout="centered")

# Apply custom CSS for theming
st.markdown(
    """
    <style>
        body, .main {
            background-color: #ffffff;  /* White background */
            color: #333;  /* Dark text color */
            font-family: Arial, sans-serif;
        }
        h1, .stTitle {
            background-color: #008000;  /* Green background for title */
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 2.3rem;
            margin: 0 auto;
        }
        .stSelectbox label, .stTextInput>div>label, .stMarkdown {
            color: #333;  /* Dark text for labels and markdown */
        }
        .stButton>button {
            background-color: #212b4a;  /* Dark blue button */
            color: white;
            border-radius: 5px;
            border: 2px solid #FF4500; /* Orange border */
        }
        .stPlotlyChart, .stAltairChart, .stBokehChart, .stDeckGlJsonChart, .stGraphvizChart, .stImage, .stVegaLiteChart {
            background-color: white;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section-header h2 {
            color: black;
            text-align: center;
            margin-top: 20px;
            font-size: 1.5rem;
        }
        .stAlert, .stMarkdown h2, .stMarkdown p {
            color: black !important; /* Ensure text is black */
            font-size: 18px !important; /* Increase font size for better visibility */
        }
        .stMarkdown h2 {
            font-weight: bold;
        }
        .custom-success {
            color: #002147; /* Red text color */
            background-color: #d4edda; /* Light green background similar to st.success */
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #28a745; /* Green border on the left */
            font-size: 18px; /* Increase font size */
        }
        .raw-data-header {
            background-color: #f0f4f8; /* Light background */
            color: #002147; /* Dark text color */
            padding: 10px;
            border-radius: 5px;
            font-size: 22px; /* Increase font size */
            font-weight: bold;
            margin-bottom: 10px; /* Space below the Raw Data header */
        }
        .stDataFrame {
            margin-top: 20px; /* Space above the table */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the page
st.markdown('<h1>Stock Price Prediction App</h1>', unsafe_allow_html=True)

st.write("""
This app predicts the **Stock Price** for selected companies using an LSTM neural network!
""")

stock_options = {
    'ICICI Bank': 'ICICIBANK.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'Infosys': 'INFY.NS',
    'HDFC Bank': 'HDFCBANK.NS',
}

selected_stock = st.selectbox('Select dataset for prediction', options=list(stock_options.keys()))

# Use buttons for chart type selection
col1, col2 = st.columns(2)

with col1:
    if st.button('Line Chart'):
        st.session_state.chart_type = 'Line Chart'

with col2:
    if st.button('Candlestick Chart'):
        st.session_state.chart_type = 'Candlestick Chart'

# Ensure the default chart type is set if no button is pressed
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'Line Chart'

chart_type = st.session_state.chart_type

@st.cache_data
def load_data(ticker):
    start_date = dt.datetime.now() - dt.timedelta(days=5*365)  # Last 5 years
    end_date = dt.datetime.now()
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data, prediction_days=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        X.append(scaled_data[i - prediction_days:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

@st.cache_resource
def build_and_train_model(X_train, y_train, stock_name, epochs=10, batch_size=32):
    model_filename = f'{stock_name}_lstm_model.h5'
    
    if os.path.exists(model_filename):
        st.markdown('<div class="custom-success">Loaded pre-trained model from cache.</div>', unsafe_allow_html=True)
        model = load_model(model_filename)
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        with st.spinner('Training the model...'):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[EarlyStopping(monitor='loss', patience=3)])
        
        model.save(model_filename)
        st.markdown('<div class="custom-success">Model trained and saved successfully.</div>', unsafe_allow_html=True)
    
    return model

def predict_future_prices(model, data, scaler, prediction_days=60, future_days=30):
    last_60_days = data['Close'].values[-prediction_days:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_prices = []
    for _ in range(future_days):
        pred_price = model.predict(X_test)
        predicted_prices.append(pred_price[0, 0])
        # Update the input with the latest prediction
        pred_price_reshaped = np.reshape(pred_price, (1, 1, 1))  # Reshape to match X_test dimensions
        X_test = np.append(X_test[:, 1:, :], pred_price_reshaped, axis=1)
    
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

def plot_line_chart(data, predicted_prices):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], label='Actual Price', color='#008000')  # Green line for actual price
    
    last_date = data['Date'].values[-1]
    future_dates = pd.date_range(start=last_date, periods=len(predicted_prices)+1, freq='B')[1:]  # Exclude the start date
    
    ax.plot(future_dates, predicted_prices, label='Predicted Price', color='red')  # Red line for predicted prices
    
    ax.set_facecolor('#f0f4f8')  # Light grey background
    ax.grid(True, which='both', linestyle='--', color='gray')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{selected_stock} Price Prediction')
    ax.legend()
    st.pyplot(fig)

def plot_candlestick_chart(data, predicted_prices):
    # Filter data to only include the last two months
    two_months_ago = data['Date'].max() - pd.DateOffset(months=2)
    recent_data = data[data['Date'] >= two_months_ago]

    last_date = recent_data['Date'].values[-1]
    future_dates = pd.date_range(start=last_date, periods=len(predicted_prices) + 1, freq='B')[1:]  # Exclude the start date

    # Create the base candlestick chart for the last two months of historical data
    candlestick = go.Candlestick(
        x=recent_data['Date'],
        open=recent_data['Open'],
        high=recent_data['High'],
        low=recent_data['Low'],
        close=recent_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Historical Data'
    )

    # Create a candlestick chart for the predicted prices (in yellow)
    predicted_candles = go.Candlestick(
        x=future_dates,
        open=predicted_prices.flatten(),
        high=predicted_prices.flatten(),
        low=predicted_prices.flatten(),
        close=predicted_prices.flatten(),
        increasing_line_color='pink',
        decreasing_line_color='yellow',
        name='Predicted Prices'
    )

    # Plot the chart
    fig = go.Figure(data=[candlestick, predicted_candles])

    # Update the layout for better visibility and to show only one label per month
    fig.update_layout(
        title=f"{selected_stock} - Candlestick Chart with Future Predictions (Last 2 Months)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        xaxis=dict(
            type='category',
            tickvals=[recent_data['Date'].dt.strftime('%Y-%m').unique()[0], 
                      recent_data['Date'].dt.strftime('%Y-%m').unique()[-1], 
                      future_dates[-1].strftime('%Y-%m')],
            ticktext=[recent_data['Date'].dt.strftime('%b %Y').unique()[0], 
                      recent_data['Date'].dt.strftime('%b %Y').unique()[-1], 
                      future_dates[-1].strftime('%b %Y')],
            tickangle=-45,  # Tilt the x-axis labels for better visibility
        ),
        yaxis=dict(
            title='Price',
            fixedrange=False
        ),
        margin=dict(l=40, r=40, t=60, b=120),  # Adjust margins to prevent clipping
        height=300,  # Increase the height of the plot
        width=600,  # Increase the width of the plot
        bargap=0.3,  # Increase gap between candlesticks for better visibility
    )

    st.plotly_chart(fig)




if st.button('Predict'):
    data = load_data(stock_options[selected_stock])
    
    # Custom styled subheader for Raw Data
    st.markdown('<div class="raw-data-header">Raw Data</div>', unsafe_allow_html=True)
    
    st.write(data.tail())
    
    prediction_days = 60
    future_days = 30  # Predict next 30 days
    
    X, y, scaler = preprocess_data(data, prediction_days)
    
    model = build_and_train_model(X, y, selected_stock)
    
    predicted_prices = predict_future_prices(model, data, scaler, prediction_days, future_days)
    
    if chart_type == 'Line Chart':
        plot_line_chart(data, predicted_prices)
    else:
        plot_candlestick_chart(data, predicted_prices)

st.markdown('<div class="footer"><h1>Created by Avani Choudhary</h1></div>', unsafe_allow_html=True)
