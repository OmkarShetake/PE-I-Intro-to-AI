import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
data = pd.read_csv("data/energy_data.csv")

X = data[['Temperature (°C)', 'Humidity (%)', 'Hour']]
y = data['Energy (kWh)']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
with open("models/energy_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/energy_model.pkl")
