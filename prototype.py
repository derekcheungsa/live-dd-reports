import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import date, datetime, timedelta
from urllib.request import urlopen
import json
import os
import certifi

fmp_key = os.environ['FMP_KEY']

def get_ratios(ticker:str, period:str="quarter"):
    url = "https://financialmodelingprep.com/api/v3/ratios/" + ticker +"?period=quarter&limit=16&apikey=" + fmp_key
    response = urlopen(url, cafile=certifi.where())
    data = json.loads(response.read().decode("utf-8"))
    data_formatted = {}
    for value in data:
        if period == "quarter":
            date = value['date'][:7]
        else:
            date = value['date'][:4]
        del value['date']
        del value['symbol']

        data_formatted[date] = value

    return pd.DataFrame(data_formatted)

print(get_ratios('TRTN'))