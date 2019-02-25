import pandas as pd
import json
import multiprocessing
import pandas as pd
from yahoo_fin import stock_info as si


class Ticker():

    def __init__(self, ticker):
        self.ticker = ticker

        self.quote = self.get_quote()
        self.i_s = self.get_financial_rec("financials")
        self.b_s = self.get_financial_rec("balance-sheet")
        self.c_f = self.get_financial_rec("cash-flow")

    def describe(self):
        print("{:s}:".format(self.ticker))
        print("Market cap is {:s} (>100 M).".format(
            self.quote["Market Cap"], self.ticker))
        print("For every {:>1.1f} (4-15) spent buying, $1 will be profited in a year.".format(
            self.quote["PE Ratio (TTM)"]))
        print("Dividend Yield is {:>s} (>7%).".format(
            self.quote["Forward Dividend & Yield"], self.ticker))

        print('done')

    def get_market_price(self):
        return si.get_live_price(self.ticker)

    def get_quote(self):
        return si.get_quote_table(self.ticker, dict_result=True)

    def get_financial_rec(self, doc_type):
        document = pd.read_html(
            'https://finance.yahoo.com/quote/{:s}/{:s}?p={:s}'.format(self.ticker, doc_type, self.ticker))
        return document
