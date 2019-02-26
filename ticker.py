import pandas as pd
import numpy as np
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

        self.shares = self.compute_outstanding_shares()

        print('done')

    def describe(self):
        print(
            "{:>5s} ==============================================================================".format(self.ticker))
        print("Market Cap is {:s} (>100 M).".format(self.quote["Market Cap"], self.ticker))

        print("For every $X (4-15) spent buying a stock, I should receive $1 in profit one year later as an owner.")

        print("Dividend Yield is {:>s} (>7%).".format(self.quote["Forward Dividend & Yield"], self.ticker))

        print('done')

    def get_market_price(self):
        return si.get_live_price(self.ticker)

    def get_quote(self):
        return si.get_quote_table(self.ticker, dict_result=True)

    def get_financial_rec(self, doc_type):
        '''
        All numbers in thousands $[[

        :param doc_type:
        :return:
        '''

        document = pd.read_html(
            'https://finance.yahoo.com/quote/{:s}/{:s}?p={:s}'.format(self.ticker, doc_type, self.ticker))

        df = document[0].dropna().replace("-", 0).T

        df.columns = df.iloc[0]
        df = df.drop(df.index[0])

        col_names = list(df)

        annual_doc = df.rename(columns={col_names[0]: 'Date', })

        convert_type = {c: 'int64' for c in list(df)}
        convert_type['Date'] = pd.datetime
        del convert_type[col_names[0]]

        annual_doc = annual_doc.astype(convert_type)

        return annual_doc

    def compute_outstanding_shares(self):
        df = self.b_s
        return df["Preferred Stock"] + df["Common Stock"] - df["Treasury Stock"]

    def compute_eps(self):
        '''
        Earnings per Share: This is probably the most important number to understand. This is the earnings per 1 share
        or the profit for 1 share. To calculate this number, divide the company’s total net income by the common shares
        outstanding.

        :return:
        '''
        # eps = self.i_s.loc['Net Income Applicable To Common Shares'] / df.loc['b']
        pass
