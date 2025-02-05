import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import logging
from pytz import timezone

# --- Set Timezone ---
tz = timezone("US/Central")

# --- Configure Logging ---
logging.basicConfig(
    filename=f"trading_strategy_{datetime.now(tz=tz)}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Parameters ---
SYMBOL = "AMZN"
STARTING_BALANCE = 1000
PCT_THRESH = 10  # bottom 10% of day's range

# Choose a historical period (adjust as needed)
start_date = "2019-01-01"
end_date = "2024-12-31"

logging.info(f"Started trading strategy - {datetime.now(tz=tz)}")

logging.info("Starting backtest for %s from %s to %s", SYMBOL, start_date, end_date)

# --- Download historical data ---
data = yf.download(SYMBOL, start=start_date, end=end_date)

# Verify necessary columns are present
if not set(["Open", "High", "Low", "Close"]).issubset(data.columns):
    logging.error("Downloaded data does not contain all OHLC columns.")
    raise ValueError("Downloaded data does not contain all OHLC columns.")

# --- Compute the trading signal ---
# A day is a signal day if the close is in the bottom PCT_THRESH% of the day's range.
data["signal"] = (data["Close"] - data["Low"]) <= (PCT_THRESH / 100) * (
    data["High"] - data["Low"]
)

# --- Backtesting the Strategy ---
balance = STARTING_BALANCE
in_trade = False
entry_price = None
entry_date = None
trades = []  # to store details of each trade

for i in range(len(data) - 1):
    today = data.iloc[i]
    tomorrow = data.iloc[i + 1]
    date_today = data.index[i]
    date_tomorrow = data.index[i + 1]

    if not in_trade:
        # Look for an entry signal
        if today["signal"]:
            entry_price = today["Close"]
            entry_date = date_today
            in_trade = True
            logging.info("Entered trade on %s at price %.2f", entry_date, entry_price)
    else:
        # Exit if today's signal condition is no longer met.
        if not today["signal"]:
            exit_price = today["Close"]
            exit_date = date_today
            trade_return = exit_price / entry_price  # Return factor for the trade
            balance *= trade_return
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_factor": trade_return,
                    "balance_after_trade": balance,
                }
            )
            logging.info(
                "Exited trade on %s at price %.2f; Trade Return: %.4f; Balance: %.2f",
                exit_date,
                exit_price,
                trade_return,
                balance,
            )
            in_trade = False
            entry_price = None
            entry_date = None

# If a trade remains open at the end of the data, exit on the final day's close.
if in_trade:
    exit_price = data.iloc[-1]["Close"]
    exit_date = data.index[-1]
    trade_return = exit_price / entry_price
    balance *= trade_return
    trades.append(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_factor": trade_return,
            "balance_after_trade": balance,
        }
    )
    logging.info(
        "Final trade exit on %s at price %.2f; Trade Return: %.4f; Balance: %.2f",
        exit_date,
        exit_price,
        trade_return,
        balance,
    )
    in_trade = False

# --- Log final results ---
logging.info("Final Balance: ${:.2f}".format(balance))
logging.info("Trade History:")
for trade in trades:
    logging.info(trade)

# --- (Optional) Plot the Equity Curve ---
# Build an equity curve that updates on days when a trade is exited.
equity_curve = pd.DataFrame(index=data.index, columns=["balance"])
equity_curve["balance"] = STARTING_BALANCE  # initialize with starting balance
current_balance = STARTING_BALANCE
trade_idx = 0

for i, current_date in enumerate(data.index):
    # If a trade exited on this date, update the balance
    if trade_idx < len(trades) and trades[trade_idx]["exit_date"] == current_date:
        current_balance = trades[trade_idx]["balance_after_trade"]
        trade_idx += 1
    equity_curve.loc[current_date, "balance"] = current_balance

plt.figure(figsize=(12, 6))
plt.plot(equity_curve.index, equity_curve["balance"], label="Equity Curve")
plt.title(f"Equity Curve for {SYMBOL} Strategy")
plt.xlabel("Date")
plt.ylabel("Account Balance ($)")
plt.legend()
plt.grid(True)
plt.show()
