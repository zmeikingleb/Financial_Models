import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import torch

#téléchargement des données
data = yf.download("EURUSD=X", start="2024-01-01", end="2025-10-01", auto_adjust=True)
data = data.asfreq('B').ffill()  # fréquence business et remplissage manquant

#calcul des taux de changement (change_rates)
change_rates = data['Close'].pct_change().dropna()

#création  de la série temporelle Darts
change_rates_df = pd.DataFrame(change_rates.values, index=change_rates.index, columns=['rates'])
series = TimeSeries.from_dataframe(change_rates_df, value_cols='rates')

#normalisation
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

#division test/train
train, test = series_scaled.split_after(0.9)

#modèle N-BEATS
model = NBEATSModel(
    input_chunk_length=300,
    output_chunk_length=5,   # prévisions lissées sur 5 jours
    n_epochs=150,
    num_blocks=3,
    num_layers=5,
    layer_widths=256,
    random_state=42
)

#GPU si disponible
if torch.cuda.is_available():
    model.to(torch.device("cuda"))

#entraînement
model.fit(train, verbose=True)

#prévision sur le test
forecast_scaled = model.predict(n=len(test), series=train)
forecast_returns = scaler.inverse_transform(forecast_scaled).to_series()

# reconstruction prix cumulés
last_price = data['Close'].iloc[len(train)-1]
forecast_price = pd.Series(index=test.time_index)
prev_price = last_price
for j in range(len(forecast_returns)):
    prev_price = prev_price * (1 + forecast_returns.iloc[j])
    forecast_price.iloc[j] = prev_price

#prévision future continue
# entraîner sur toute la série (train + test) pour prédiction future
full_series_scaled = scaler.fit_transform(series)
model.fit(full_series_scaled, verbose=True)

future_horizon = 30  # nombre de jours futurs
future_scaled = model.predict(n=future_horizon, series=full_series_scaled)
future_returns = scaler.inverse_transform(future_scaled).to_series()

# reconstruire prix futur à partir du dernier prix réel
last_known_price = data['Close'].iloc[-1]
future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_horizon, freq='B')
future_prices = pd.Series(index=future_index)
prev_price = last_known_price
for j in range(len(future_returns)):
    prev_price = prev_price * (1 + future_returns.iloc[j])
    future_prices.iloc[j] = prev_price

#visualisation
real_price = data['Close'].iloc[len(train):]

plt.figure(figsize=(12, 6))
plt.plot(real_price, label='Actual Close')
plt.plot(forecast_price, label='Forecast Close')
plt.plot(future_prices, label='Future Forecast', linestyle='--', color='orange')
plt.title("EUR/USD Forecast with N-BEATS")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#calcul du MAPE sur test
ts_real = TimeSeries.from_dataframe(pd.DataFrame(real_price.values, index=real_price.index, columns=['price']), value_cols='price')
ts_forecast = TimeSeries.from_dataframe(pd.DataFrame(forecast_price.values, index=forecast_price.index, columns=['price']), value_cols='price')
error = mape(ts_real, ts_forecast)
print(f"MAPE: {error:.2f}%")
