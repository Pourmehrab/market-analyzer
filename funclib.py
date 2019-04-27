import pandas as pd
import numpy as np
import json
from multiprocessing import Process, Manager
from yahoo_fin.stock_info import *
from sklearn import linear_model
import statsmodels.api as sm
import wolframalpha
import pprint
import matplotlib.pyplot as plt
import math
from IPython.display import display, clear_output


def intrinsic_val_calc(ticker, MARR):
    tick = {'ticker': ticker, }
    print("started {:s} ...".format(ticker))
    log = ''
    try:
        income_statement, balance_sheet, cash_flow = get_fin_statements(ticker)

        bv_flow = get_book_val(balance_sheet)

        if len(bv_flow) == 0:
            log += 'No data were available to use. '
            return None

        log = check_incom_stat(income_statement, log)
        log = check_bal_sheet(balance_sheet, log)
        log = check_cash_flow(cash_flow, log)

        # tick['old book value'], tick['curr book value'] = book_val[0], book_val[-1]
        # tick['yrs in between'] = len(book_val)

        bv_ror = bv_rate_of_ret(bv_flow)
        if bv_ror < 0:
            log += 'BV DECLINED. '

        stats = convert2dict(get_stats(ticker))
        raw_div = stats['Trailing Annual Dividend Yield 3']
        div_rate = float(raw_div.replace('%', '')) / 100 if isinstance(raw_div, str) else 0
        curr_bv = bv_flow[-1]
        intrinsic_val = buss_intrinsic_val(curr_bv, div_rate, bv_ror, MARR)
        bv_per_share = float(stats['Book Value Per Share (mrq)'])
        value_per_share = bv_per_share * intrinsic_val / curr_bv

        market_price = get_live_price(ticker)

        disc_rate = (value_per_share - market_price) / market_price
        log = check_stats(stats, log)

        tick['BV G rate'] = bv_ror
        tick['BV'] = bv_per_share
        tick['P'] = market_price
        tick['Val'] = value_per_share
        tick['disc'] = disc_rate
        tick['note'] = log

        return tick
    except():
        return None


def check_stats(stats, log):
    val = stats['Trailing P/E']
    if isinstance(val, str) and float(val) > 15:
        log += 'High P/E. '
    val = stats['Return on Equity (ttm)']
    if isinstance(val, str) and float(val.replace('%', '').replace(',', '')) < 10:
        log += 'low ROE. '
    val = stats['Total Debt/Equity (mrq)']
    if isinstance(val, str) and float(val) > 0.5:
        log += 'High D/E. '
    val = stats['Current Ratio (mrq)']
    if isinstance(val, str) and float(val) < 1.5:
        log += 'Low Curr Ratio. '

    return log


def check_incom_stat(income_statement, log):
    income = income_statement['Net Income From Continuing Ops'][0]
    if income < 0:
        log += 'Income was -{:s}. '.format(scale_number(-income))
    _, rev_rate = do_ols(income_statement['Total Revenue'][:])
    if rev_rate < 0:
        log += 'Rev declined {:s}/yr. '.format(scale_number(-rev_rate))

    return log


def check_bal_sheet(balance_sheet, log):
    _, asset_rate = do_ols(balance_sheet['Total Current Assets'][:])
    if asset_rate < 0:
        log += 'Assets declined {:s}/yr. '.format(scale_number(-asset_rate))
    _, liabilities_rate = do_ols(balance_sheet['Total Current Liabilities'][:])
    if liabilities_rate > 0:
        log += 'Liabilities increased {:s}/yr. '.format(scale_number(liabilities_rate))

    return log


def check_cash_flow(cash_flow, log):
    _, cash_rate = do_ols(cash_flow['Total Cash Flow From Operating Activities'][:])
    if cash_rate < 0:
        log += 'Cash declined {:s}/yr. '.format(scale_number(-cash_rate))

    _, invest_rate = do_ols(cash_flow['Total Cash Flows From Investing Activities'][:])
    if invest_rate > 0:
        log += 'Investment declined {:s}/yr. '.format(scale_number(invest_rate))

    _, fin_rate = do_ols(cash_flow['Total Cash Flows From Financing Activities'][:])
    if fin_rate > 0:
        log += 'Finances declined {:s}/yr. '.format(scale_number(fin_rate))

    return log


def scale_number(n):
    n_dig = math.floor(math.log(n, 10) + 1)
    if n_dig > 9:
        return str(round(n / 10 ** 9, 1)) + ' B'
    elif n_dig > 6:
        return str(round(n / 10 ** 6, 1)) + ' M'
    else:
        return str(round(n / 10 ** 3, 1)) + ' T'


def isnan(x):
    return x != x


def convert_str_percent(s):
    if isnan(s):
        return 0
    else:
        return float(s.strip('%')) / 100


def bv_rate_of_ret(bv_flow):
    '''
    Solve -BV_0 + BV_n/(1+r)^n = 0 for r

    :param bv_flow: cash flow of BVs
    :return: The rate that the bv grew in years
    '''

    return min((bv_flow[-1] / bv_flow[0]) ** (1 / len(bv_flow)) - 1, 0.18)


def buss_intrinsic_val(curr_bv, div_rate, bv_growth_rate, MARR, tax=.22):
    n = len(MARR)
    fut_bv_flow = np.array([curr_bv * (1 + bv_growth_rate) ** i for i in range(1, n + 1)])
    dividend = sum(np.array(
        [fut_bv_flow[i - 1] * div_rate / (1 + MARR[i - 1]) ** i for i in range(1, n + 1)]))

    return (dividend + fut_bv_flow[-1] * (1 - tax) / (1 + MARR[- 1]) ** n) / (1 + tax / (1 + MARR[- 1]) ** n)


def get_book_val(balance_sheet):
    '''
    the order is from [past ... present]
    '''
    temp = balance_sheet['Total Assets'] - balance_sheet['Total Liabilities']
    temp = temp[temp.iloc[:].apply(lambda x: x != '-')]

    return np.flip(temp.iloc[:].values, axis=0)


def get_fin_statements(ticker):
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
    return raw_data[0].iloc[1:, 0].values.tolist()


def get_yield_curve(yr, save=False):
    raw_data = pd.read_html(
        'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')
    years = raw_data[1].iloc[0, 1:]
    rates = np.array(list(raw_data[1].iloc[-1, 1:].astype(float)))

    for i in range(len(years)):
        plt.text(i, rates[i], str(rates[i]), verticalalignment='center')
    if save:
        fig = plt.plot(years, rates)
        plt.yticks([])
        plt.show()
        # plt.savefig('yield_curve.pdf', format='pdf')

    return rates[:yr] / 100


def do_ols(book_val):
    Y = book_val
    X = np.array([i + 1 for i in range(len(Y))], dtype=int)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print(model.summary())

    return model.rsquared, model.params[1]