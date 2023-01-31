import requests
import time
import talib
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
import json
from sklearn.model_selection import train_test_split

# Set the API endpoint URL and the ticker symbol for bitcoin
ENDPOINT_URL = "https://api.binance.com/api/v3/ticker/24hr"
TICKER_SYMBOL = "BTCUSDT"

# Set the number of past days to use as input data
NUM_PAST_DAYS = 365

# Set up sell/buy rules
THRESHOLD = 0.005
HOLD_PERIOD = 1


# Preprocessing function to normalize and scale the input data
def preprocess_data(X):
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_scaled = X_norm / np.max(np.abs(X_norm), axis=0)
    return X_scaled

# Function to retrieve additional data from internet and social media news sources


def retrieve_news_data():
    # Make an HTTP GET request to the news website
    response = requests.get("https://news.bitcoin.com")
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the relevant information from the website
    titles = [title.text for title in soup.find_all("h3")]
    descriptions = [desc.text for desc in soup.find_all("p")]
    links = [link.get("href") for link in soup.find_all("a")]

    # Return the extracted information
    return {"titles": titles, "descriptions": descriptions, "links": links}

# Function to perform natural language processing on news articles

def nlp_preprocess_news(news_data):
    # Placeholder for now, you can add code here to perform NLP on the news data
    processed_news_data = [5, 6, 7, 8]
    return processed_news_data


# Build the model
model = tf.keras.Sequential()

# Add layers to the model to handle multiple input features
# Change the input_dim to include the number of additional data points from the news sources
input_dim = NUM_PAST_DAYS*4+4+4+4
model.add(tf.keras.layers.Dense(64, input_dim=input_dim, activation="relu"))
model.add(tf.keras.layers.Dense(64, input_dim=input_dim, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(1))

# Compile the model with a learning rate of 0.001 and a batch size of 32
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)


# Initialize lists to store the input and target data for training
X_train = []
Y_train = []

# Initialize lists to store the input and target data for testing
X_test = []
Y_test = []


# creates empty lists to be used to store the corresponding data for each day.

open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []


while True:
    # Send the API request to retrieve the current price and volume data for bitcoin
    response = requests.get(ENDPOINT_URL, params={"symbol": TICKER_SYMBOL})
    data = response.json()


    # Extract the current open, high, low, and close prices and volume from the API response
    current_open = data["openPrice"]
    current_high = data["highPrice"]
    current_low = data["lowPrice"]
    current_close = data["lastPrice"]
    current_volume = data["quoteVolume"]


    # Append the current open, high, low, and close prices and volume to the X_train and Y_train list
    X_train.append([current_open, current_high, current_low,
                   current_close, current_volume])
    Y_train.append([current_close])

    # Add the current open, high, low, and close prices and volume to the list of past prices and volumes

    open_prices.append(current_open)
    high_prices.append(current_high)
    low_prices.append(current_low)
    close_prices.append(current_close)
    volumes.append(current_volume)

    # If the lists of past prices and volumes have more than NUM_PAST_DAYS elements, remove the oldest elements
    if len(open_prices) > NUM_PAST_DAYS:
        open_prices.pop(0)
        high_prices.pop(0)
        low_prices.pop(0)
        close_prices.pop(0)
        volumes.pop(0)

    # Calculate technical indicators from the past NUM_PAST_DAYS days' worth of price data
    sma_50 = talib.SMA(np.array(close_prices).astype(np.double), timeperiod=50)
    sma_200 = talib.SMA(np.array(close_prices, dtype=float), timeperiod=200)
    rsi = talib.RSI(np.array(close_prices, dtype=float), timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        np.array(close_prices, dtype=float), fastperiod=12, slowperiod=26, signalperiod=9)

    # Retrive News Data and then process it.
    news_data = retrieve_news_data()
    processed_news_data = nlp_preprocess_news(news_data)

        # Get the length of the shortest array
    min_len = min([len(open_prices), len(high_prices), len(low_prices), len(close_prices), len(volumes), len(sma_50), len(sma_200), len(rsi), len(macd), len(macd_signal), len(macd_hist), len(processed_news_data)])

    # Slice all arrays to have the same length as the shortest array
    open_prices = open_prices[:min_len]
    high_prices = high_prices[:min_len]
    low_prices = low_prices[:min_len]
    close_prices = close_prices[:min_len]
    volumes = volumes[:min_len]
    sma_50 = sma_50[:min_len]
    sma_200 = sma_200[:min_len]
    rsi = rsi[:min_len]
    macd = macd[:min_len]
    macd_signal = macd_signal[:min_len]
    macd_hist = macd_hist[:min_len]
    processed_news_data = processed_news_data[:min_len]

    # Stack the arrays into a single array
    X = np.column_stack((open_prices, high_prices, low_prices, close_prices, volumes, sma_50, sma_200, rsi, macd, macd_signal, macd_hist))
   
    # Normalize and scale the input data
    X_scaled = preprocess_data(X)

    # Retrieve the most recent target value (the next day's close price)
    Y = close_prices[-1]


    # Split the data into a training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

    # Preprocess the input data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # Train the model on the training data
    model.fit(X_train, Y_train, batch_size=32, epochs=50)

       # Evaluate the model on the testing data
    test_loss = model.evaluate(X_test, Y_test)
    print("Test loss:", test_loss)

    # Use the model to make predictions on new data
    predictions = model.predict(X_test)

       # Decide whether to buy or sell based on the predictions and the threshold
    for i in range(len(predictions)):
            if predictions[i] > THRESHOLD:
                print("Buy at", close_prices[i])
            elif predictions[i] < -THRESHOLD:
                print("Sell at", close_prices[i])
            else:
                print("Hold")

        # Wait for HOLD_PERIOD days before making the next prediction
            time.sleep(HOLD_PERIOD * 24 * 60 * 60)
