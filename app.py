import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import date, datetime, timedelta
from urllib.request import urlopen
import json
import os
import tweepy
from fastapi import FastAPI,Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#Use this dict to add term to the query, you can use OR in the string for more than one term
#
query_parameters = {'AR': '#natgas',
                    'FLNG': '#lng',
                    'SDE': '$SDE.TO'}

app = FastAPI()
app.mount("/assets", StaticFiles(directory="public/assets"))


# your Twitter API credentials
consumer_key = os.environ['TWITTER_KEY']
consumer_secret = os.environ['TWITTER_SECRET']
access_token  = os.environ['TWITTER_ACCESS_TOKEN'] 
access_token_secret  = os.environ['TWITTER_ACCESS_SECRET'] 
fmp_key = os.environ['FMP_KEY']

templates = Jinja2Templates(directory="public/templates")



# Main code needed to render the get the tweets and render in HTML
@app.get("/tweet/{symbol_name}", response_class=HTMLResponse)
async def tweet(symbol_name: str, request: Request):
    # create an OAuth1 authentication object 
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)

    # create a Tweepy API client
    api = tweepy.API(auth)
    max_tweets = 150
    symbol_name_upper = symbol_name.upper()
   
    term = query_parameters.get(symbol_name_upper,'None')
    if (term == 'None'):
        query = "$"+symbol_name_upper
    else:
        query = "$"+symbol_name_upper + " OR " + term

    # Use the search/tweets endpoint to retrieve tweets matching the search term
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(max_tweets)
    
    # Custom the filter you want to use for twitter search
    # This needs clean-up, I filtered for $AR and removed the references to bitcoin related for this symbol
    #
    filtered_tweet =[]
    for tweet in tweets:
        if ("crypto" in tweet.user.name.lower()):
            continue
        if ("crypto" in tweet.full_text.lower()):
            continue
        if ("digital" in tweet.full_text.lower()):
            continue
        if ("coin" in tweet.full_text.lower()):
            continue
        if ("solana" in tweet.full_text.lower()):
            continue
        if ("arweave" in tweet.full_text.lower()):
            continue
        if ("blockchain" in tweet.full_text.lower()):
            continue
        if (tweet.author.followers_count > 1000):
            filtered_tweet.append(tweet)

    return templates.TemplateResponse(f"{symbol_name_upper}.html", {"request":request, "tweets": filtered_tweet})

# The code below shows how to surface info to a front-end (such as GSheets) via REST API
#
#
option_chain_cols = [
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
]

option_chain_dict = {"openInterest": "openinterest", "impliedVolatility": "iv"}

def get_full_option_chain(symbol: str) -> pd.DataFrame:
    """Get all options for given ticker [Source: Yahoo Finance]

    Parameters
    ----------
    symbol: str
        Stock ticker symbol

    Returns
    -------
    pd.Dataframe
        Option chain
    """
    ticker = yf.Ticker(symbol)
    dates = ticker.options

    options = pd.DataFrame()

    for _date in dates:
        calls = ticker.option_chain(_date).calls
        puts = ticker.option_chain(_date).puts
        calls = calls[option_chain_cols].rename(columns=option_chain_dict)
        puts = puts[option_chain_cols].rename(columns=option_chain_dict)
        calls.columns = [x + "_c" if x != "strike" else x for x in calls.columns]
        puts.columns = [x + "_p" if x != "strike" else x for x in puts.columns]
        temp = pd.merge(calls, puts, how="outer", on="strike")
        temp["expiration"] = _date
        options = pd.concat([options, temp], axis=0).reset_index(drop=True)

    return options

def get_put_call_ratio(
    symbol: str,
    window: int = 30,
    start_date: str = None,
) -> pd.DataFrame:
    """Gets put call ratio over last time window [Source: AlphaQuery.com]

    Parameters
    ----------
    symbol: str
        Ticker symbol to look for
    window: int, optional
        Window to consider, by default 30
    start_date: str, optional
        Start date to plot  (e.g., 2021-10-01), by default last 366 days

    Returns
    -------
    pd.DataFrame
        Put call ratio

    Examples
    --------
    >>> from openbb_terminal.sdk import openbb
    >>> pcr_df = openbb.stocks.options.pcr("B")
    """

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d")

    url = f"https://www.alphaquery.com/data/option-statistic-chart?ticker={symbol}\
        &perType={window}-Day&identifier=put-call-ratio-volume"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/70.0.3538.77 Safari/537.36"
    }

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()

    pcr = pd.DataFrame.from_dict(r.json())
    pcr.rename(columns={"x": "Date", "value": "PCR"}, inplace=True)
    pcr.set_index("Date", inplace=True)
    pcr.index = pd.to_datetime(pcr.index).tz_localize(None)

    return pcr[pcr.index > start_date]


@app.get("/get_analyst/{symbol_name}")
async def get_analyst(symbol_name: str):
    t = yf.Ticker(symbol_name)
    info = t.info
    return {"Analysts": info['numberOfAnalystOpinions']}

@app.get("/get_rsi/{symbol_name}")
async def get_rsi(symbol_name: str):
    ticker = yf.download(symbol_name)
    ticker = ticker.dropna()
    rsi_values = ta.rsi(ticker['Close'].tail(50)).tail(30)  
    return {rsi_values.to_json(date_unit="s", date_format="iso")}

@app.get("/get_rsi_scalar/{symbol_name}")
async def get_rsi_scalar(symbol_name: str):
    ticker = yf.download(symbol_name)
    ticker = ticker.dropna()
    rsi_values = ta.rsi(ticker['Close'].tail(50)).tail(1)  
    return {rsi_values.to_json(date_unit="s", date_format="iso")}

@app.get("/get_pcr/{symbol_name}")
async def get_pcr(symbol_name: str):
    pcr_df=get_put_call_ratio(symbol_name).tail(120)
    return {pcr_df.to_json(date_unit="s", date_format="iso")}

@app.get("/get_pcr_scalar/{symbol_name}")
async def get_pcr_scalar(symbol_name: str):
    pcr_df=get_put_call_ratio(symbol_name).tail(1)
    return {pcr_df.to_json(date_unit="s", date_format="iso")}

@app.get("/get_ratios_ttm/{symbol_name}")
async def get_ratios_ttm(symbol_name: str):
    response = urlopen("https://financialmodelingprep.com/api/v3/ratios-ttm/"+symbol_name+"?apikey="+fmp_key)
    data = json.loads(response.read().decode("utf-8"))
    return data

@app.get("/get_key_metrics_ttm/{symbol_name}")
async def get_key_metrics_ttm(symbol_name: str):
    response = urlopen("https://financialmodelingprep.com/api/v3/key-metrics-ttm/"+symbol_name+"?apikey"+fmp_key)
    data = json.loads(response.read().decode("utf-8"))
    return data

@app.get("/get_call_option_chain/{symbol_name}")
async def get_option_chain(symbol_name: str):
    options_data = get_full_option_chain(symbol_name)
    options_data = options_data.dropna()
    options_data=options_data.drop(columns=['lastTradeDate_p','lastPrice_p','bid_p','ask_p','volume_p', 'iv_p','openinterest_p'])
    options_data=options_data[options_data['openinterest_c'] > 50]
    options_data=options_data[options_data['bid_c'] > 0]
    options_data.sort_values(by='iv_c', inplace=True, ascending=False)
    options_data['lastTradeDate_c'] = pd.to_datetime(options_data['lastTradeDate_c'])
    options_data['lastTradeDate_c'] = options_data['lastTradeDate_c'].dt.date
    
    today = date.today()
    weekday = date.weekday(today)
    if weekday == 5 or weekday == 6:
        print ("weekday " + str(weekday))
        friday = today - timedelta(days=today.weekday() - 5)
        filtered_df = options_data[options_data['lastTradeDate_c'] == friday]
    else:
        filtered_df = options_data[options_data['lastTradeDate_c'] == today]

    if (filtered_df.empty):
       filtered_df= options_data.head(100)
    
    return {filtered_df.to_json(date_unit="s", date_format="iso",orient="values")}

@app.get("/get_dividend/{symbol_name}")
async def get_dividend(symbol_name: str):
    df=yf.Ticker(symbol_name).dividends.tail(1)
    return {df.to_json(date_unit="s", date_format="iso",orient="values")}


@app.get("/tweet/{symbol_name}", response_class=HTMLResponse)
async def tweet(symbol_name: str, request: Request):

    # create an OAuth1 authentication object
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)

    # create a Tweepy API client
    api = tweepy.API(auth)
    max_tweets = 150
    symbol_name_upper = symbol_name.upper()

    term = query_parameters.get(symbol_name,'None')
    if (term == 'None'):
        query = "$"+symbol_name
    else:
        query = "$"+symbol_name + " OR " + term

    # Use the search/tweets endpoint to retrieve tweets matching the search term
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(max_tweets)
    
    # Custom the filter you want to use for twitter search
    filtered_tweet =[]
    for tweet in tweets:
        if ("crypto" in tweet.user.name.lower()):
            continue
        if ("crypto" in tweet.full_text.lower()):
            continue
        if ("digital" in tweet.full_text.lower()):
            continue
        if ("coin" in tweet.full_text.lower()):
            continue
        if ("solana" in tweet.full_text.lower()):
            continue
        if ("arweave" in tweet.full_text.lower()):
            continue
        if ("blockchain" in tweet.full_text.lower()):
            continue
        if (tweet.author.followers_count > 1000):
            filtered_tweet.append(tweet)

    return templates.TemplateResponse(f"{symbol_name_upper}.html", {"request":request, "tweets": filtered_tweet})