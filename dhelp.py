""" Seeking Alpha Model """
__docformat__ = "numpy"

import logging
import requests
import pandas as pd

from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import lambda_long_number_format

logger = logging.getLogger(__name__)

# Customize 
def get_exchange_dict () :
    return { 'TSLA' : 'NASDAQ',
                  'GOOG' : 'NASDAQ',
                  'MSFT' : 'NASDAQ',
                  'GLNG' : 'NASDAQ',
                  'BBBY' : 'NASDAQ',
                  'TMDX' : 'NASDAQ'
            }

def get_similar_companies_dict():
    return {'AM' : ['EPD','ET','ENB','PBA','MPLX'],
            'AR': ['RRC', 'EQT','SWN','CNX','CHK'],
            'FLNG' : ['GLNG','SFL'],
            'FTCO': ['GOLD','KGC','AU','AEM','NEM'],
            'GSL' : ['DAC','CMRE','SFL'],
            'MP' : ['SGML','LAC','MTRN'],
            'INSW' : ['FRO','TRMD','EURN'],
            'IBM' : ['MSFT','GOOGL','INTC','HPQ','AAPL'],
            'MPLX' : ['AM','EPD','ENB','PBA','ET'],
            'TRTN' : ['TGH','AER','GATX'],
            'KNTK' : ['AM','EPD','ENB','PBA','ET','MPLX'],
            'V' : ['MA','PYPL','SQ','EBAY','FIS'],
            'ZIM' : ['MATX']
            }

def get_investor_report_url_dict():
    return {'TRTN': 'https://www.tritoninternational.com/sites/triton-corp/files/investor-presentation-nov-2022.pdf',
            'AM'  : 'https://d1io3yog0oux5.cloudfront.net/_374edef9c4170f864475079b2fb421fd/anteromidstream/db/711/6478/pdf/AM+Website+Presentation+December+2022_vF2_11.30.22.pdf',
            'GSL': 'https://drive.google.com/file/d/1ZJAgOEVbZKH96mahev-FGMRpUui0Lym8/preview', 
            'NS' : 'https://investor.nustarenergy.com/static-files/67e67e05-d236-4ccf-8ee3-78a2d93e57a4',
            'GLNG': 'https://www.golarlng.com/~/media/Files/G/Golar-Lng/documents/presentation/golar-lng-limited-2022-q3-results-presentation.pdf',
            'MP' : 'https://s25.q4cdn.com/570172628/files/doc_presentations/2022/11/MP-3Q22-Earnings-Deck-FINAL.pdf',
            'TMDX': 'https://investors.transmedics.com/static-files/c4f69c45-77b0-4981-a5a7-b404ab4aae95',
            'FLNG': 'https://ml-eu.globenewswire.com/Resource/Download/362a3031-2be8-4698-93d0-8066618b20b6',
            'JXN': 'https://s28.q4cdn.com/568090435/files/doc_presentation/Analyst-Day-Presentation.pdf',         
            'CMRE': 'https://drive.google.com/file/d/1Hz4B8nDCK_oJoiEEdPy2q_aqXPxZOwx_/preview',                
            'EGY' : 'https://d1io3yog0oux5.cloudfront.net/_202883002163863943d602098d2b6e88/vaalco/db/776/7755/pdf/November+IR+Deck+Final+v1.pdf',
            'EPR' : 'https://investors.eprkc.com/investor-presentation/default.aspx',
            'MSFT': 'https://view.officeapps.live.com/op/view.aspx?src=https://c.s-microsoft.com/en-us/CMSFiles/SlidesFY22Q2.pptx?version=4a9117e1-04ff-5143-12a0-1ecfc041aa59',
            'AROC': 'https://s26.q4cdn.com/362558937/files/doc_presentations/2022/11/AROC-Investor-Presentation_RBCWidescreen-vFinal.pdf',
            'V'   : 'https://s29.q4cdn.com/385744025/files/doc_downloads/2022/Visa-Inc-Fiscal-2022-Annual-Report.pdf',
            'TSLA': 'https://tesla-cdn.thron.com/static/SVCPTV_2022_Q4_Quarterly_Update_JZPPNX.pdf?xseo=&response-content-disposition=inline%3Bfilename%3D%22TSLA-Q4-2022-Update.pdf%22',
            'AR' : 'https://d1io3yog0oux5.cloudfront.net/_6cc4d707f40f7cc56aebe36b019cb270/anteroresources/db/641/5970/pdf/AR+Investor+Presentation_Dec_12.01.2022_vF3.pdf'
            }

def get_morningstar_report_url_dict():
    return {'TSLA': 'https://drive.google.com/file/d/1Hppn9KbAXpg-44Z1MGy-e1LXyZQheLXn/preview',
            'TRTN': 'https://drive.google.com/file/d/1Hppn9KbAXpg-44Z1MGy-e1LXyZQheLXn/preview',
            'MP' :  'https://drive.google.com/file/d/1X30f9SFsY7dlGSyl-7i1QkItEi-QZBCY/preview',
            'MSFT': 'https://drive.google.com/file/d/13Ay0BFGV-3RuES6Q1Ak92kKoLgHbAO3k/preview'
            }

def get_historical_metric(ticker: str, metric:str) -> pd.DataFrame:
    url = "https://seekingalpha.com/api/v3/symbol_data/charting"

    querystring = {
      "end": "2023-02-04",
      "metrics": metric,
      "slugs[]" : ticker,
      "start":   "2018-01-30",
    }

    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
        "Accept": "*/*",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
    }

    response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring
    )

    seek_object = response.json()["data"][ticker.lower()][metric]
    dict = eval(seek_object)
    df = pd.DataFrame({'values': list(dict.values())}, index=dict.keys())
    #output = pd.DataFrame.from_dict(seek_object)
    #output.index = ["Date", metric ]
    print(df)


@log_start_end(log=logger)
def get_estimates_eps(ticker: str, quarterly: bool) -> pd.DataFrame:
    """Takes the ticker, asks for seekingalphaID and gets eps estimates

    Parameters
    ----------
    ticker: str
        ticker of company
    Returns
    -------
    pd.DataFrame
        eps estimates for the next 10yrs
    Examples
    --------
    >>> from openbb_terminal.sdk import openbb
    >>> openbb.stocks.fa.epsfc("AAPL")
    """

    url = "https://seekingalpha.com/api/v3/symbol_data/estimates"
    if (quarterly):
        time_period = "quarterly"
    else:
        time_period = "annual"

    querystring = {
        "estimates_data_items": "eps_normalized_actual,eps_normalized_consensus_low,eps_normalized_consensus_mean,"
        "eps_normalized_consensus_high,eps_normalized_num_of_estimates",
        "period_type": time_period,
        "relative_periods": "-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11",
    }

    # add ticker_ids for the ticker
    seek_id = get_seekingalpha_id(ticker)
    querystring["ticker_ids"] = seek_id

    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
        "Accept": "*/*",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
    }

    # semi random user agent -- disabled and static user agent because it might be the reason for 403
    # headers["User-Agent"] = get_user_agent()

    response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring
    )

    # init
    output = pd.DataFrame(
        columns=[
            "fiscalyear",
            "consensus_mean",
            "change %",
            "analysts",
            "actual",
            "consensus_low",
            "consensus_high",
        ]
    )

    # if no estimations exist, response is empty "reviews" and "reviews"
    # {"revisions":{},"estimates":{}}
    try:
        seek_object = response.json()["estimates"][str(seek_id)]

        items = len(seek_object["eps_normalized_num_of_estimates"].keys())

        for i in range(0, items - 3):
            # python_dict
            eps_estimates = {}
            eps_estimates["fiscalyear"] = seek_object[
                "eps_normalized_num_of_estimates"
            ][str(i)][0]["period"]["fiscalyear"]
            eps_estimates["analysts"] = seek_object["eps_normalized_num_of_estimates"][
                str(i)
            ][0]["dataitemvalue"]
            try:
                eps_estimates["actual"] = seek_object["eps_normalized_actual"][str(i)][
                    0
                ]["dataitemvalue"]
            except Exception:
                eps_estimates["actual"] = 0
            eps_estimates["consensus_low"] = seek_object[
                "eps_normalized_consensus_low"
            ][str(i)][0]["dataitemvalue"]
            eps_estimates["consensus_high"] = seek_object[
                "eps_normalized_consensus_high"
            ][str(i)][0]["dataitemvalue"]
            eps_estimates["consensus_mean"] = seek_object[
                "eps_normalized_consensus_mean"
            ][str(i)][0]["dataitemvalue"]

            try:
                this = float(eps_estimates["consensus_mean"])
                try:
                    prev = float(
                        seek_object["eps_normalized_actual"][str(i - 1)][0][
                            "dataitemvalue"
                        ]
                    )
                except Exception:
                    prev = float(
                        seek_object["eps_normalized_consensus_mean"][str(i - 1)][0][
                            "dataitemvalue"
                        ]
                    )

                percent = ((this / prev) - 1) * 100
            except Exception:
                percent = 0

            eps_estimates["change %"] = percent

            # format correction (before return, so calculation still works)
            eps_estimates["consensus_mean"] = lambda_long_number_format(
                float(eps_estimates["consensus_mean"])
            )
            eps_estimates["consensus_low"] = lambda_long_number_format(
                float(eps_estimates["consensus_low"])
            )
            eps_estimates["consensus_high"] = lambda_long_number_format(
                float(eps_estimates["consensus_high"])
            )
            eps_estimates["actual"] = lambda_long_number_format(
                float(eps_estimates["actual"])
            )

            # df append replacement
            new_row = pd.DataFrame(eps_estimates, index=[0])
            output = pd.concat([output, new_row])
    except Exception:
        return pd.DataFrame()

    return output


@log_start_end(log=logger)
def get_estimates_rev(ticker: str) -> pd.DataFrame:
    """Takes the ticker, asks for seekingalphaID and gets rev estimates

    Parameters
    ----------
    ticker: str
        ticker of company
    Returns
    -------
    pd.DataFrame
        rev estimates for the next 10yrs
    Examples
    --------
    >>> from openbb_terminal.sdk import openbb
    >>> openbb.stocks.fa.revfc("AAPL")
    """

    url = "https://seekingalpha.com/api/v3/symbol_data/estimates"

    querystring = {
        "estimates_data_items": "revenue_actual,revenue_consensus_low,revenue_consensus_mean,"
        "revenue_consensus_high,revenue_num_of_estimates",
        "period_type": "annual",
        "relative_periods": "-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11",
    }

    # add ticker_ids for the ticker
    seek_id = get_seekingalpha_id(ticker)
    querystring["ticker_ids"] = seek_id

    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
        "Accept": "*/*",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Connection": "keep-alive",
        "TE": "trailers",
    }

    # semi random user agent -- disabled and static user agent because it might be the reason for 403
    # headers["User-Agent"] = get_user_agent()

    response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring
    )

    # init
    # pd.empty should deliver true if no data-rows are added
    output = pd.DataFrame(
        columns=[
            "fiscalyear",
            "consensus_mean",
            "change %",
            "analysts",
            "actual",
            "consensus_low",
            "consensus_high",
        ]
    )

    # if no estimations exist, response is empty "reviews" and "reviews"
    # {"revisions":{},"estimates":{}}
    try:
        seek_object = response.json()["estimates"][seek_id]

        items = len(seek_object["revenue_num_of_estimates"].keys())

        for i in range(0, items - 3):
            # python_dict
            revenue_estimates = {}
            revenue_estimates["fiscalyear"] = seek_object["revenue_num_of_estimates"][
                str(i)
            ][0]["period"]["fiscalyear"]
            revenue_estimates["consensus_mean"] = seek_object["revenue_consensus_mean"][
                str(i)
            ][0]["dataitemvalue"]

            revenue_estimates["analysts"] = seek_object["revenue_num_of_estimates"][
                str(i)
            ][0]["dataitemvalue"]
            if i < 1:
                revenue_estimates["actual"] = seek_object["revenue_actual"][str(i)][0][
                    "dataitemvalue"
                ]
            else:
                revenue_estimates["actual"] = 0
            revenue_estimates["consensus_low"] = seek_object["revenue_consensus_low"][
                str(i)
            ][0]["dataitemvalue"]
            revenue_estimates["consensus_high"] = seek_object["revenue_consensus_high"][
                str(i)
            ][0]["dataitemvalue"]

            try:
                this = float(revenue_estimates["consensus_mean"])
                # if actual revenue is available, take it for the calc
                try:
                    prev = float(
                        seek_object["revenue_actual"][str(i - 1)][0]["dataitemvalue"]
                    )
                except Exception:
                    prev = float(
                        seek_object["revenue_consensus_mean"][str(i - 1)][0][
                            "dataitemvalue"
                        ]
                    )

                percent = ((this / prev) - 1) * 100
            except Exception:
                percent = float(0)

            revenue_estimates["change %"] = percent

            # format correction (before return, so calculation still works)
            revenue_estimates["consensus_mean"] = lambda_long_number_format(
                float(revenue_estimates["consensus_mean"])
            )
            revenue_estimates["consensus_low"] = lambda_long_number_format(
                float(revenue_estimates["consensus_low"])
            )
            revenue_estimates["consensus_high"] = lambda_long_number_format(
                float(revenue_estimates["consensus_high"])
            )
            revenue_estimates["actual"] = lambda_long_number_format(
                float(revenue_estimates["actual"])
            )

            # df append replacement
            new_row = pd.DataFrame(revenue_estimates, index=[0])
            output = pd.concat([output, new_row])
    except Exception:
        return pd.DataFrame()

    return output


@log_start_end(log=logger)
def get_seekingalpha_id(ticker: str) -> str:
    """Takes the ticker, asks for seekingalphaID and returns it

    Parameters
    ----------
    ticker: str
        ticker of company
    Returns
    -------
    str:
        seekingalphaID - to be used for further API calls
    """

    url = "https://seekingalpha.com/api/v3/searches"

    querystring = {
        "filter[type]": "symbols",
        "filter[list]": "all",
        "page[size]": "1",
    }

    querystring["filter[query]"] = ticker
    payload = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
        "Accept": "*/*",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://seekingalpha.com/",
        "Connection": "keep-alive",
        # "TE": "trailers",
    }

    response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring
    )

    try:
        seekingalphaID = str(response.json()["symbols"][0]["id"])
    except Exception:
        # for some reason no mapping possible
        seekingalphaID = "0"

    return seekingalphaID


