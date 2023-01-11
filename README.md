## 1. Purpose
The purpose of this project is to use the power of OpenBB SDK to help make investment decisions systematically and thoroughly.
This project builds upon the due diligence jupiter notebook that comes with the SDK and adds to it integrations such as to Twitter, 
Tradingview real-time widgets, and Google Sheets so that there are additional insights that can help with investment decisions.



## 2. Getting started


   **OpenBB Due Diligence Jupiter Notebook:**

   This notebook is the one that does the analysis and will generate the html code after it is run

   1. First install the OpenBB SDK following the [SDK instructions](https://github.com/OpenBB-finance/OpenBBTerminal/blob/develop/README.md).  
   2. The due_diligence.ipynb notebook contains the code that gathers the financial data and generates the HTML file.  
   3. Go into the notebook and specify the symbol and long name for the symbol 
   4. Running the report will produce an html file <symbol-name>.html in the public/templates directory


   **Backend:**
   The backend is very simple.  It uses FastAPI and only has one main routine that handles the serving of the html page.  
   You can deploy this in one of many Clouds.  As an example, you can deploy on render.com

   1. Select a python 3.9 image
   2. Specify the build command to be build.sh
   3. Add in the Twitter API keys in the environment variables (TWITTER_KEY,TWITTER_SECRET,TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
   4. Add in the deploy command
   uvicorn app:app --host 0.0.0.0 --port 10000


## 3.License
Distributed under MIT license