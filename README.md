# AI Binance Market Assistant

A Streamlit educational market-analysis app. It fetches public Binance spot candles, calculates technical indicators, and trains a Random Forest Classifier on historical features.

## Files
- `app.py` — application
- `requirements.txt` — dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Streamlit:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open https://share.streamlit.io/ and connect your GitHub account.
3. Select the repository and `app.py` as the main file.
4. Deploy. No Binance API key is required for public market candles.

## How to use
- Enter a Binance spot market such as `BTC/USDT`, `ETH/USDT`, or `SOL/USDT`.
- Select a timeframe and number of historical candles.
- Click **Analyze market**.
- Review the candlestick chart, technical indicators, model scenarios and holdout score.

## Important limitations
- This is an educational prototype, not financial advice.
- Random Forest learns from historical labels; it cannot reliably predict the future or guarantee profit.
- The displayed holdout accuracy is a simple chronological evaluation, not a full trading backtest and not proof of profitability.
- The latest candle may still be forming.
- The app does not place trades and does not need private exchange credentials.
- Before relying on any strategy, build a robust walk-forward backtest that accounts for fees, slippage, liquidity and data leakage.
