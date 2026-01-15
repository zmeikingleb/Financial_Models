# Financial Models – USD/EUR Exchange Rate Forecasting

## Objective
The goal of this project is to use the **N-BEATS model** to forecast the **trend of the USD/EUR exchange rate**.

## Data Source
Historical exchange rate data is obtained using the **`yfinance`** library in Python.

## Methodology
1. A **time series** is created from the historical data using the `TimeSeries` class from the **Darts** Python library.
2. The time series is **normalized** to ensure consistent and coherent results.
3. **Price forecasting** is performed using the following formula:
$P_t = P_{t-1} \times (1 + r_t)$

where $r_t$ are the **direct returns** forecast by N-BEATS.

4. The model is trained on **5 days** of historical data to forecast the **next 30 days**.

## Notes
- The focus is on short-term forecasting of the USD/EUR exchange rate.  
- Normalization of the time series is crucial for the model to perform accurately.



