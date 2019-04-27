from funclib import *

if __name__ == "__main__":
    """
    It uses the following factors to decided on composition of portfolio:
    
     1) analyst judgment after reading the reports
     2) budget with taxation info
     3) scenarios (inflation, tax, security performance)
     4) financial statements (runs checks)
     5) treasury yield curve
     6) 
    
    Question to answer: Should I buy companiy X's share?
        Compare the Minimum Atractive Rate of Return (MARR) against the treasury bill of 
        whatever number of years you're intending to hold. If higher, buy.
    
    What is the current worth of share of company X?
        Estimate the present value of estimated cashflow and determine the discounted price by 25%.
        If price is lower, buy.
    
    
    
    """
    n_holding_yrs = 5  # number of yrs intending to hold after buying
    MARR = get_yield_curve(n_holding_yrs)

    tick = intrinsic_val_calc("CL", MARR)
    column_names = tick.keys()

    data_dic = {c: [tick[c], ] for c in column_names}

    ticker_sp500_list = pull_sp500_tickers()

    t_list = ticker_sp500_list  # ticker_list
    for i, t in enumerate(t_list):
        tick = intrinsic_val_calc(t, MARR)
        if tick is not None:
            for c in column_names:
                data_dic[c].append(tick[c])
            clear_output(wait=True)
            print(i + 1, "/", len(t_list))

    df = pd.DataFrame.from_dict(data_dic)
    # df.to_csv('analysis_picked.csv', index=False)
    df.to_csv('analysis_sp500.csv', index=False)
