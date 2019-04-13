import pandas as pd
import numpy as np
import json
import multiprocessing
from yahoo_fin.stock_info import *
from sklearn import linear_model
import statsmodels.api as sm
import wolframalpha
import pprint
import mechanize
import matplotlib.pyplot as plt
import xlrd


def lets_do_it(ticker):
    tick = {'ticker': ticker, }

    try:
        income_statement, balance_sheet, cash_flow = get_fundamental_docs(ticker)
        book_val = get_book_val(balance_sheet)
        if len(book_val) < 1:
            return None

        tick['old book value'], tick['curr book value'] = book_val[0], book_val[-1]
        tick['yrs in between'] = len(book_val)

        tick['4yr book value rate'] = bv_4yr_rate(book_val)

        stats = convert2dict(get_stats(ticker))
        tick['trailing annual dividend yield'] = convert_str_percent(stats['Trailing Annual Dividend Yield 3'])
        tick['avg rate of return'] = tick['4yr book value rate'] + tick['trailing annual dividend yield']

        tick['profit margin'] = convert_str_percent(stats['Profit Margin'])

        tick['price/book (mrq)'] = float(stats['Price/Book (mrq)'])
        tick['4yr book r2'], tick['4yr book slope'] = book_val_growth(book_val)

        tick['market price'] = get_live_price(ticker)
        tick['4yr avg debt ratio'] = compute_avg_debt_ratio(balance_sheet)

        return tick
    except():
        return None


def isnan(x):
    return x != x


def convert_str_percent(s):
    if isnan(s):
        return 0
    else:
        return float(s.strip('%')) / 100


def compute_avg_debt_ratio(balance_sheet):
    return balance_sheet['Total Liabilities'][0] / balance_sheet['Total Stockholder Equity'][0]


def bv_4yr_rate(book_val):
    initial_cap = book_val[0]
    cash_flow = book_val - initial_cap
    cash_flow[0] -= initial_cap

    return -np.irr(cash_flow)


def get_book_val(balance_sheet):
    '''
    the order is from [past ... present]
    '''
    temp = balance_sheet['Total Assets'] - balance_sheet['Total Liabilities']
    return np.flip(temp.iloc[:].values)


def get_fundamental_docs(ticker):
    income_statement = refine(get_income_statement(ticker))
    balance_sheet = refine(get_balance_sheet(ticker))
    cash_flow = refine(get_cash_flow(ticker))

    return income_statement, balance_sheet, cash_flow


def refine(df):
    df = df.dropna().replace("-", "0").T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    convert_type = {c: 'float' for c in list(df)}
    return df.astype(convert_type) * 1000


def convert2dict(df):
    return {df['Attribute'][i]: df['Value'][i] for i in range(len(df))}


def pull_sp500_tickers():
    raw_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return raw_data[0].iloc[1:, 1]


def get_yield_curve(yr):
    raw_data = pd.read_html(
        'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')
    years = raw_data[1].iloc[0, 1:]
    rates = list(raw_data[1].iloc[-1, 1:].astype(float))
    fig = plt.plot(years, rates)
    # plt.show()
    plt.yticks([])

    for i in range(len(years)):
        plt.text(i,rates[i],str(rates[i]),verticalalignment='center')
    plt.savefig('yield_curve.pdf', format='pdf')

    return float(raw_data[1].iloc[-1, yr]) / 100


def book_val_growth(book_val):
    Y = book_val
    X = np.array([i + 1 for i in range(len(Y))], dtype=int)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print(model.summary())

    return model.rsquared, model.params[1]


if __name__ == "__main__":
    """
    It uses the following factors to decided on composition of portfolio:
    
     1) analyst judgment after reading the reports
     2) budget with taxation info
     3) scenarios (inflation, tax, security performance)
     4) financial statements (runs checks)
     5) treasury yield curve
     6) 
     
    
    
    """
    rate = get_yield_curve(10)

    # client = wolframalpha.Client('ULQLV8-TQ3HG44Q7R')
    # res = client.query('p/e atvi')
    # print(res)

    tick = lets_do_it("ATVI")
    column_names = tick.keys()
    print('done.')

    dic = {c: [tick[c], ] for c in column_names}

    ticker_list = pull_sp500_tickers()
    n = len(ticker_list)
    for i, t in enumerate(ticker_list):
        tick = lets_do_it(t)
        if tick is not None:
            for c in column_names:
                dic[c].append(tick[c])
            print(i, "/", n)

    df = pd.DataFrame.from_dict(dic)
    df.to_csv('analysis.csv', index=False)
    print("done!")
