import pandas as pd
from ticker import Ticker

def pull_sp500_tickers():
    raw_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')


if __name__ == "__main__":

    ticker = Ticker("atvi")
    ticker.describe()

    print("done!")
