import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('insurance.csv')

print("Original data: ")
print(df.head())

df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("Encoded data: ")
print(df_encoded.head())

y = df_encoded['charges']
X = df_encoded.drop('charges', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train test split complete")

model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Performance")
print(f"mean absolute error: {mae:.2f}")
print(f"mean squared error: {mse:.2f}")
print(f"r2 score: {r2:.4f}")

intercept = model.intercept_
coefficients = model.coef_

coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

print(f"Base Price (intercept): {intercept:.2f}")
print(f"Feature Coefficients")
print(coef_df.sort_values(by='Coefficient', ascending=False))

plt.figure(figsize=(8,6))

sns.scatterplot(x=y_test, y=predictions, color = 'blue', alpha = 0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = 'red', linestyle='--', linewidth=2)
plt.xlabel("Actual Charges ($) ")
plt.ylabel("Predicted Charges ($) ")
plt.title("Actual vs Predicted Insurance Charges")

plt.show()