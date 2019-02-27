import pandas as pd
import numpy as np
import json
import multiprocessing
import pandas as pd
from yahoo_fin.stock_info import *
from sklearn import linear_model
import statsmodels.api as sm


# http://theautomatic.net/yahoo_fin-documentation/#get_analysts_info


class Ticker():

    def __init__(self, ticker):
        self.ticker = ticker

        self.income_statement = self.refine(get_income_statement(ticker))
        self.balance_sheet = self.refine(get_balance_sheet(ticker))
        self.cash_flow = self.refine(get_cash_flow(ticker))
        self.market_price = get_live_price(ticker)
        self.quote = get_quote_table(ticker, dict_result=True)
        self.stats = self.convert2dict(get_stats(ticker))

        self.shares = self.income_statement['Net Income Applicable To Common Shares'][0] / \
                      float(self.stats['Diluted EPS (ttm)'])

        temp = self.balance_sheet['Total Assets'] - self.balance_sheet['Total Liabilities']
        self.book_val = temp.iloc[:].values

        self.r2, self.slope = self.stability()

        print('done')

    @staticmethod
    def refine(df):
        df = df.dropna().replace("-", "0").T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        convert_type = {c: 'int64' for c in list(df)}
        return df.astype(convert_type) * 10 ** 3

    @staticmethod
    def convert2dict(df):
        return {df['Attribute'][i]: df['Value'][i] for i in range(len(df))}

    def describe(self):
        print(
            "{:>5s} ==============================================================================".format(self.ticker))
        print("Market Cap is {:s} (>100 M).".format(self.quote["Market Cap"], self.ticker))

        print("For every $X (4-15) spent buying a stock, I should receive $1 in profit one year later as an owner.")

        print("Dividend Yield is {:>s} (>7%).".format(self.quote["Forward Dividend & Yield"], self.ticker))

        print("The 4-yr book value grew at {:>2.1f} M, r2 is {:>d} (>95).".format(self.slope, self.r2))

        print('done')

    def stability(self):
        Y = self.book_val
        X = np.array([len(Y) - i for i in range(len(Y))], dtype=int)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        # print(model.summary())

        return int(model.rsquared * 100), model.params[1] / 10 ** 6
