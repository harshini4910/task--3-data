import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error


data = pd.read_csv(
    r"C:\Users\Suresh S\OneDrive\Desktop\task 3\historical_data.csv",
    parse_dates=['Date'],
    index_col='Date'
)


data.index = pd.to_datetime(data.index)
data = data.asfreq('MS')


train = data.iloc[:-12]
test = data.iloc[-12:]


model = ARIMA(train['Value'], order=(5,1,0))
model_fit = model.fit()


forecast = model_fit.forecast(steps=12)

plt.figure(figsize=(10,5))
plt.plot(train.index, train['Value'], label='Train')
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.legend()
plt.show()

mae = mean_absolute_error(test['Value'], forecast)
print("Mean Absolute Error:", mae)
