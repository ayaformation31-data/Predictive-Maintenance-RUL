import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_data, add_rul, drop_useless_columns


# 1. Loading data
data = load_data("data/raw/train_FD001.txt")

# 2. Adding RUL
data = add_rul(data)

# 3. Cleaning the data
data = drop_useless_columns(data)

# 4. Split by engines
units = data['unit'].unique()
train_units, test_units = train_test_split(units, test_size=0.2, random_state=42)

train_data = data[data['unit'].isin(train_units)]
test_data = data[data['unit'].isin(test_units)]

# 5. X / y
X_train = train_data.drop(columns=['unit', 'time', 'RUL'])
y_train = train_data['RUL']

X_test = test_data.drop(columns=['unit', 'time', 'RUL'])
y_test = test_data['RUL']

# 6. Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Assessment
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)