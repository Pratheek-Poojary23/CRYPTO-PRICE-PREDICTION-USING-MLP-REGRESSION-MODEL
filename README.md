# Ethereum Price Prediction Using MLP(Multilayer perceptron) Regression Models



### Problem Statement

This study aims to forecast Ethereum (ETH) prices in Indian Rupees (INR) using MLP 
Regression, emphasizing dimension and feature engineering methodologies. The primary 
challenge is to develop accurate predictive models that anticipate Ethereum price movements, 
including changes in price signs, to support informed investment decisions.

### Data

A single dataset for Ethereum (ETH) prices in Indian Rupees (INR), 
collected from Kaggle. The dataset spans from January 2018 to July 2021 and contains essential 
attributes such as date, open price, high price, low price, close price, adjusted close price, and 
trading volume.

### Feature engineering

- The selected features used in our model include 'Year', 'Month', 'Day', 'Open', 'High', and 'Low'. 
These features represent temporal aspects (year, month, day) as well as daily price fluctuations 
(open, high, low) of Ethereum.
- The 'Adj Close' column, representing adjusted closing prices, is excluded as a feature in our analysis. Adjusted closing prices are typically adjusted for factors such as dividends, stock splits or other corporate actions, which may not directly reflects ETH's market demand.

```python
data = data.drop(['Adj Close'], axis=1)
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low']]
y = data['Close']
```

The 'Date' column in the dataset contains timestamps representing the date and time when each 
observation was recorded. Extracting the year, month, and day from the 'Date' column allows 
the model to capture potential seasonal or yearly patterns in Ethereum price movements. 

```python
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

```

### Model Implementation: 
- The dataset is split into training and testing sets, with 80% allocated for training and 20% for testing.
```python
train_size = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

```
Notably, the split was done sequentially to preserve the temporal sequence of the data. Random splitting may bias the evaluation of model performance, especially if there are systematic changes in the data over time. 

- To ensure model stability and convergence, features are standardized using the StandardScaler to achieve zero mean and unit variance.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
```

- The MLP regressor is instantiated with specific hyperparameters, in our implementation, two hidden layers are configured with 100 and 50 neurons, and the "ReLU" activation function is utilized. The "Adam" solver is employed for optimization, with a maximum of 1000 iterations allowed for model convergence.
```python
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_regressor.fit(X_train_scaled, y_train_scaled)
```
- The MLP regressor is trained on the scaled training data to learn the relationship between the input features and the target variable. Predictions are subsequently made on the scaled test set.
```python
y_pred = mlp_regressor.predict(X_test_scaled)
y_pred_train=mlp_regressor.predict(X_train_scaled)
```
