import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import sys
import os

# Get last downloaded symbol from file
last_symbol_path = "data/last_symbol.txt"
default_symbol = "IDEA"
if os.path.exists(last_symbol_path):
    try:
        with open(last_symbol_path, "r") as f:
            default_symbol = f.read().strip()
    except:
        pass

# Get symbol from command line argument or use last downloaded symbol
symbol = sys.argv[1].upper() if len(sys.argv) > 1 else default_symbol
url = f"https://stockanalysis.com/quote/nse/{symbol}/history/"
csv_path = "data/raw.csv"

headers = {"User-Agent": "Mozilla/5.0"}

# Fetch page
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "lxml")

table = soup.find("table")
if table is None:
    print("[ERROR] Failed to find data table.")
    exit()

rows = table.find("tbody").find_all("tr")

data = []
for row in rows:
    cols = row.find_all("td")
    if len(cols) < 6:
        continue

    try:
        date = datetime.strptime(cols[0].text.strip(), "%b %d, %Y")
        open_price = float(cols[1].text.replace(",", "").strip())
        high_price = float(cols[2].text.replace(",", "").strip())
        low_price = float(cols[3].text.replace(",", "").strip())
        close_price = float(cols[4].text.replace(",", "").strip())
        volume_text = cols[6].text.replace(",", "").strip() if len(cols) > 6 else "0"
        volume = int(volume_text) if volume_text.isdigit() else 0

        data.append([date, open_price, high_price, low_price, close_price, volume])
    except:
        continue

df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

# Filter last 90 days up to today
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
df = df[(df.index >= start_date) & (df.index <= end_date)]

# Indicators
df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["Returns"] = df["Close"].pct_change()

# Remove warmup rows
df = df.iloc[20:]

df.reset_index(inplace=True)
df.to_csv(csv_path, index=False)

# Save the current symbol as the last downloaded symbol
with open(last_symbol_path, "w") as f:
    f.write(symbol)

print("[SUCCESS] Saved last 90 days data (up to today) to:", csv_path)
print(f"[DATE] Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"[DATA] Total rows: {len(df)}")
print(f"[SYMBOL] Symbol: {symbol}")
print(df.tail())
