import pandas as pd
from ticker import Ticker

dataSP500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')


ticker = Ticker("ATVI")
q = ticker.get_quote()
p = ticker.get_market_price()
ticker.describe()
print("done!")
