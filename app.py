
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AI Crypto Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Crypto Prediction Dashboard")
st.caption("Educational predictions using historical cryptocurrency prices.")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")

symbol = st.sidebar.text_input(
    "Crypto symbol",
    value="BTC/USDT"
).strip().upper()

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["15m", "1h", "4h", "1d"]
)

candle_count = st.sidebar.slider(
    "Historical candles",
    min_value=300,
    max_value=1000,
    value=500,
    step=100
)

horizon = st.sidebar.selectbox(
    "Prediction horizon",
    ["Next candle", "Next 3 candles"]
)

# Yahoo Finance uses symbols such as BTC-USD instead of BTC/USDT.
def yahoo_symbol(user_symbol):
    value = user_symbol.upper().strip()

    if "/" in value:
        base, quote = value.split("/", 1)
        if quote in ["USDT", "USD", "USDC"]:
            return f"{base}-USD"

    if value.endswith("USDT"):
        return value[:-4] + "-USD"

    if value.endswith("USD") and not value.endswith("-USD"):
        return value[:-3] + "-USD"

    return value


@st.cache_data(ttl=300)
def load_data(ticker, selected_timeframe, count):
    if selected_timeframe == "15m":
        interval = "15m"
        period = "60d"
    elif selected_timeframe in ["1h", "4h"]:
        interval = "1h"
        period = "730d"
    else:
        interval = "1d"
        period = "max"

    data = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data is None or data.empty:
        raise ValueError(
            "No market data returned. Check the symbol or try another timeframe."
        )

    # Handle Yahoo Finance's possible multi-level column names.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Market data is missing columns: {missing}")

    data = data[required].copy()
    data = data.dropna()

    if selected_timeframe == "4h":
        data = data.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    return data.tail(count).copy()


# -------------------------------
# Load market data
# -------------------------------
ticker = yahoo_symbol(symbol)

try:
    with st.spinner(f"Loading historical data for {ticker}..."):
        df = load_data(ticker, timeframe, candle_count)

except Exception as error:
    st.error(f"Could not load market data: {error}")
    st.info(
        "Try BTC/USDT with 1h or 1d first. "
        "Yahoo Finance availability depends on the coin and timeframe."
    )
    st.stop()

if len(df) < 100:
    st.error("Not enough historical data. Try a different timeframe or coin.")
    st.stop()

# -------------------------------
# Technical indicators
# -------------------------------
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

relative_strength = avg_gain / avg_loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + relative_strength))

df["MACD"] = (
    df["close"].ewm(span=12, adjust=False).mean()
    - df["close"].ewm(span=26, adjust=False).mean()
)

df["MACD_signal"] = df["MACD"].ewm(
    span=9, adjust=False
).mean()

df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(20).std()
df["volume_change"] = df["volume"].pct_change()

# Target: direction of the next candle.
df["future_return"] = df["close"].shift(-1) / df["close"] - 1

# 0 = downward, 1 = sideways, 2 = upward
df["target"] = np.select(
    [
        df["future_return"] < -0.002,
        df["future_return"] > 0.002
    ],
    [0, 2],
    default=1
)

feature_columns = [
    "RSI",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "MACD",
    "MACD_signal",
    "return",
    "volatility",
    "volume_change"
]

model_data = df.dropna(
    subset=feature_columns + ["target"]
).copy()

# The final row has no known next-candle target.
training_data = model_data.iloc[:-1]

if len(training_data) < 80:
    st.error("Not enough clean data to train the model.")
    st.stop()

X = training_data[feature_columns]
y = training_data["target"].astype(int)

# Time-ordered split to reduce look-ahead leakage.
split_index = int(len(training_data) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

if y_train.nunique() < 2:
    st.error("The training data has too few target classes.")
    st.stop()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

accuracy = None
if len(X_test) > 0:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

# Latest feature row for the prediction.
latest_features = model_data[feature_columns].iloc[[-1]]
prediction = int(model.predict(latest_features)[0])

class_names = {
    0: "Downward",
    1: "Sideways",
    2: "Upward"
}

class_emojis = {
    0: "🔴",
    1: "🟡",
    2: "🟢"
}

probabilities = model.predict_proba(latest_features)[0]
class_probabilities = dict(zip(model.classes_, probabilities))

# -------------------------------
# Dashboard
# -------------------------------
latest_close = float(df["close"].iloc[-1])
previous_close = float(df["close"].iloc[-2])
price_change = (
    (latest_close - previous_close) / previous_close * 100
    if previous_close else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest close (USD)",
    f"${latest_close:,.4f}",
    f"{price_change:+.2f}%"
)

col2.metric(
    "Model direction",
    f"{class_emojis[prediction]} {class_names[prediction]}"
)

col3.metric(
    "RSI",
    f"{df['RSI'].iloc[-1]:.2f}"
)

if accuracy is not None:
    col4.metric("Test accuracy", f"{accuracy:.1%}")
else:
    col4.metric("Test accuracy", "N/A")

st.caption(
    f"Data source: Yahoo Finance ({ticker}). "
    "Prices are quoted in USD, not USDT."
)

st.warning(
    "This is an experimental machine-learning estimate, not a guaranteed "
    "price forecast or financial advice. Model accuracy on historical data "
    "does not guarantee future performance."
)

# -------------------------------
# Candlestick chart
# -------------------------------
st.subheader("📊 Historical Price Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["SMA_20"],
    name="SMA 20",
    line=dict(width=1)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["SMA_50"],
    name="SMA 50",
    line=dict(width=1)
))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Indicator"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Model probabilities
# -------------------------------
st.subheader("🤖 Model Prediction")

prob_cols = st.columns(3)

for col, class_id in zip(prob_cols, [2, 1, 0]):
    probability = class_probabilities.get(class_id, 0.0)
    col.metric(
        f"{class_emojis[class_id]} {class_names[class_id]} probability",
        f"{probability:.1%}"
    )

st.write(
    f"**Selected horizon:** {horizon}. "
    "The classifier predicts the direction of the next candle; "
    "the 3-candle option does not train a separate three-candle model."
)

# -------------------------------
# Technical indicator interpretation
# -------------------------------
st.subheader("📌 Technical Indicator Summary")

rsi = float(df["RSI"].iloc[-1])
macd = float(df["MACD"].iloc[-1])
macd_signal = float(df["MACD_signal"].iloc[-1])
sma20 = float(df["SMA_20"].iloc[-1])
sma50 = float(df["SMA_50"].iloc[-1])

if rsi >= 70:
    st.write("**RSI:** Above 70, often described as overbought.")
elif rsi <= 30:
    st.write("**RSI:** Below 30, often described as oversold.")
else:
    st.write("**RSI:** Between 30 and 70.")

if macd > macd_signal:
    st.write("**MACD:** Above its signal line.")
else:
    st.write("**MACD:** Below its signal line.")

if sma20 > sma50:
    st.write("**Moving averages:** SMA 20 is above SMA 50.")
else:
    st.write("**Moving averages:** SMA 20 is below SMA 50.")

# -------------------------------
# Feature importance
# -------------------------------
st.subheader("🧠 Model Feature Importance")

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# -------------------------------
# Recent market data
# -------------------------------
with st.expander("View recent historical data"):
    st.dataframe(df.tail(20), use_container_width=True)

st.caption(
    "Predictions are uncertain. Always verify market information "
    "independently and never rely on this model alone."
)
