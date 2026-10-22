import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Crypto Market Assistant", page_icon="📈", layout="wide")

st.title("📈 AI Crypto Market Assistant")
st.caption("Educational market analysis using public data (Yahoo Finance, Kraken fallback) and a Random Forest model.")

st.warning(
    "Educational tool only — not financial advice. Crypto is volatile. "
    "Model estimates are uncertain and are not guarantees. This app does not place trades."
)

# ---------------------------------------------------------------------------
# Data loading (public, no API key, not region-restricted like Binance)
# ---------------------------------------------------------------------------
YAHOO_CONFIG = {
    # timeframe: (period, yahoo interval, resample rule)
    "15m": ("60d", "15m", None),
    "1h": ("730d", "1h", None),
    "4h": ("730d", "1h", "4h"),
    "1d": ("5y", "1d", None),
}


def to_yahoo_symbol(symbol: str) -> str:
    base, quote = symbol.split("/")
    if quote in ("USDT", "USDC", "BUSD"):
        quote = "USD"
    return f"{base}-{quote}"


def to_kraken_symbol(symbol: str) -> str:
    base, quote = symbol.split("/")
    if quote in ("USDT", "USDC", "BUSD"):
        quote = "USD"
    return f"{base}/{quote}"


def fetch_yahoo(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    period, interval, resample = YAHOO_CONFIG[timeframe]
    df = yf.download(
        to_yahoo_symbol(symbol), period=period, interval=interval,
        auto_adjust=True, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned no candles.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df.columns = [str(c).lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    if resample:
        df = (
            df.set_index("timestamp")
            .resample(resample)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "close"])
            .reset_index()
        )
    return df.tail(limit).reset_index(drop=True)


def fetch_kraken(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    exchange = ccxt.kraken({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(to_kraken_symbol(symbol), timeframe=timeframe, limit=min(limit, 720))
    if not rows:
        raise ValueError("Kraken returned no candles.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    errors = []
    for name, fn in (("Yahoo Finance", fetch_yahoo), ("Kraken", fetch_kraken)):
        try:
            df = fn(symbol, timeframe, limit)
            if len(df) > 0:
                df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]})
                return df, name
        except Exception as exc:  # try next source
            errors.append(f"{name}: {exc}")
    raise ValueError("All public data sources failed. " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data["sma_10"] = close.rolling(10).mean()
    data["sma_20"] = close.rolling(20).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["return_1"] = close.pct_change()
    data["return_3"] = close.pct_change(3)
    data["volatility_10"] = data["return_1"].rolling(10).std()
    vol = data["volume"].replace(0, np.nan)  # avoid division by zero -> inf
    data["volume_change"] = vol.pct_change(fill_method=None)
    data["sma_gap"] = (close / data["sma_20"]) - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    # Tame extreme outliers so they never overflow float32 inside scikit-learn.
    data["volume_change"] = data["volume_change"].clip(-5, 5)
    for col in ["return_1", "return_3", "sma_gap"]:
        data[col] = data[col].clip(-1, 1)
    return data


FEATURES = [
    "rsi", "return_1", "return_3", "volatility_10",
    "volume_change", "sma_gap", "macd", "macd_signal",
]

with st.sidebar:
    st.header("Market settings")
    coin = st.text_input(
        "Coin / market symbol", value="BTC/USDT",
        help="Examples: BTC/USDT, ETH/USDT, SOL/USDT (USDT is mapped to USD automatically).",
    )
    timeframe = st.selectbox("Candle timeframe", ["15m", "1h", "4h", "1d"], index=1)
    candle_limit = st.slider("Historical candles", min_value=300, max_value=1000, value=700, step=100)
    horizon = st.selectbox("Prediction horizon", ["Next candle", "Next 3 candles"], index=1)
    run = st.button("Analyze market", type="primary", use_container_width=True)

if not run:
    st.info("Enter a market symbol in the sidebar, choose a timeframe, then click **Analyze market**.")
    st.markdown("**Data:** Yahoo Finance (public), with Kraken public API as fallback. No API key needed.")
    st.markdown("**AI model:** RandomForestClassifier, trained on recent historical candles each time you analyze.")
    st.stop()

symbol = coin.strip().upper()
if "/" not in symbol:
    symbol = f"{symbol}/USDT"

horizon_steps = 1 if horizon == "Next candle" else 3

try:
    with st.spinner(f"Fetching {symbol} market data and training the model..."):
        raw, source = fetch_ohlcv(symbol, timeframe, candle_limit)
        data = add_features(raw)

        # Label each candle by whether the close is higher after the selected horizon.
        data["future_return"] = data["close"].shift(-horizon_steps) / data["close"] - 1
        threshold = 0.001  # neutral band
        data["target"] = np.select(
            [data["future_return"] > threshold, data["future_return"] < -threshold],
            [1, -1], default=0,
        )

        usable = data.dropna(subset=FEATURES + ["future_return"]).copy()
        usable = usable.iloc[:-horizon_steps].copy() if horizon_steps else usable

        # Final safety net: keep only rows where every feature is finite.
        X_all = usable[FEATURES].astype("float64")
        finite_mask = np.isfinite(X_all.to_numpy()).all(axis=1)
        usable = usable.loc[finite_mask]
        X = usable[FEATURES].astype("float64")
        y = usable["target"].astype(int)

        if len(usable) < 150:
            raise ValueError("Not enough usable historical data. Increase the candle count or choose another market.")

        # Chronological holdout: never shuffle time-series data.
        split = int(len(usable) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if y_train.nunique() < 2:
            raise ValueError("Training data has only one class. Try another timeframe or more candles.")

        model = RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        accuracy = (
            accuracy_score(y_test, model.predict(X_test.to_numpy())) if len(X_test) else float("nan")
        )

        feat_rows = data.dropna(subset=FEATURES)
        latest_features = feat_rows.iloc[[-1]][FEATURES].astype("float64")
        if not np.isfinite(latest_features.to_numpy()).all():
            raise ValueError("Latest candle has invalid indicator values. Try again later.")
        probabilities = model.predict_proba(latest_features.to_numpy())[0]
        class_prob = {int(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)}
        predicted_class = int(model.predict(latest_features.to_numpy())[0])
        latest = data.iloc[-1]
except Exception as exc:
    st.error(f"Could not analyze **{symbol}**. Check the market symbol and your internet connection.")
    st.code(str(exc))
    st.stop()

st.caption(f"Data source: {source}")

label_names = {-1: "Downward scenario", 0: "Sideways / small move", 1: "Upward scenario"}
label_colors = {-1: "🔴", 0: "🟡", 1: "🟢"}
prob_down = class_prob.get(-1, 0.0)
prob_side = class_prob.get(0, 0.0)
prob_up = class_prob.get(1, 0.0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{latest['close']:,.8g} USD")
c2.metric("RSI (14)", f"{latest['rsi']:.1f}" if pd.notna(latest["rsi"]) else "N/A")
c3.metric("Model scenario", f"{label_colors[predicted_class]} {label_names[predicted_class]}")
c4.metric(
    "Chronological holdout accuracy", f"{accuracy:.1%}" if pd.notna(accuracy) else "N/A",
    help="Historical holdout score only; it does not predict future accuracy.",
)

st.subheader("1. Price chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data["timestamp"], open=data["open"], high=data["high"],
    low=data["low"], close=data["close"], name="Candles",
))
fig.add_trace(go.Scatter(x=data["timestamp"], y=data["sma_20"], name="SMA 20", line=dict(width=1.5)))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark",
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Model scenario estimates")
prob_df = pd.DataFrame({
    "Scenario": ["Upward", "Sideways / small move", "Downward"],
    "Model estimate": [prob_up, prob_side, prob_down],
})
st.bar_chart(prob_df.set_index("Scenario"))
st.caption("These are model class estimates from historical patterns, not calibrated odds or a promise of price direction.")

st.subheader("3. Indicator explanation")
rsi = latest["rsi"]
if pd.isna(rsi):
    rsi_text = "RSI is not available for the latest candle."
elif rsi >= 70:
    rsi_text = "RSI is above 70, which is commonly described as an overbought reading; it is not a sell signal by itself."
elif rsi <= 30:
    rsi_text = "RSI is below 30, which is commonly described as an oversold reading; it is not a buy signal by itself."
else:
    rsi_text = f"RSI is {rsi:.1f}, between the commonly watched 30 and 70 levels."

st.write(f"- **RSI:** {rsi_text}")
st.write(f"- **MACD:** {'above' if latest['macd'] > latest['macd_signal'] else 'below'} its signal line in the latest candle.")
st.write(f"- **20-candle average:** latest close is {'above' if latest['close'] > latest['sma_20'] else 'below'} the SMA 20.")
st.write(f"- **Latest candle timestamp:** {latest['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}.")

st.subheader("4. What to do next")
st.write(
    "Use this page to study the market, then verify the signal with your own research. "
    "Wait for a candle close before interpreting a completed-candle signal. "
    "Do not treat the model output as a guaranteed entry, target, or stop-loss."
)
st.info(
    "This starter app intentionally does not generate personalized buy/sell instructions, "
    "calculate guaranteed entry prices, or place real orders. Test any strategy with realistic "
    "fees and slippage in a separate backtest or paper-trading simulator."
)

with st.expander("Show latest market data"):
    st.dataframe(
        data[["timestamp", "open", "high", "low", "close", "volume", "rsi", "macd"]].tail(30),
        use_container_width=True,
    )
