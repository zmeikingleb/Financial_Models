# Financial_Models
The aim of this work would be to use the NBEATS predictor to forecast the tendency of the exchange rate between USD/EUR. 

The historical data used would come frome the `yfinance` library in Python. 

In order to study this prediction, a time series is to be created. That can be made via `TimeSeries` class of `darts` Python library. This series has to be normalized, so that coherent results could be obtained.

To forecast prices, the following formula is used : 
$P_{t} = P_{t-1} \times (1+r_t)$

Where $r_t$ are direct returns, forecast using NBEATS.

The data is forecast on 30 days, using NBEATS trained on 5 days.




