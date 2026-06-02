# 🏥 Health Insurance Cost Prediction: Linear Regression

##  Project Overview
This project implements a Machine Learning pipeline using **Linear Regression** to predict medical insurance costs based on patient data. The objective is to understand how different variables (like age, BMI, and smoking habits) impact insurance charges and to build a predictive model evaluating these relationships.

Through this project, I practically explored Simple vs. Multiple Linear Regression, handled categorical data encoding, and interpreted model coefficients to extract business insights.

##  Technologies & Libraries Used
* **Python 3.x**
* **Pandas:** Data preprocessing and one-hot encoding
* **Scikit-Learn (sklearn):** Model building, train-test splitting, and performance evaluation metrics
* **Matplotlib & Seaborn:** Data visualization and regression line plotting

##  Pipeline Steps & Key Insights

### 1. Data Preprocessing & Encoding
The dataset contained categorical text columns (`sex`, `smoker`, `region`). To make this data readable for the mathematical ML model, I applied **One-Hot Encoding** (using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity/the dummy variable trap).

![Encoded Data](Screenshots/Encoded%20Data.png)

### 2. Model Training & Evaluation
The data was split into 80% Training and 20% Testing sets. A Multiple Linear Regression model was trained and evaluated using standard regression metrics.
* **R² Score (~0.78):** Indicates that our features explain approximately 78% of the variance in the insurance charges, which is a strong performance for a linear model.

![Model Performance](Screenshots/Model%20Performance.png)

### 3. Interpreting Feature Coefficients
By extracting the model's coefficients, I derived actual business logic from the mathematics:
* **The Smoker Penalty:** Being a smoker increases the predicted medical charges by a massive **~$23,651**.
* **Age & BMI Impact:** Every additional year of age adds ~$256 to the bill, and every 1-point increase in BMI adds ~$337.

![Feature Coefficients](Screenshots/Feature%20Coefficients.png)

### 4. Visualizing the Results (Actual vs Predicted)
To visually evaluate how well the model performed, I plotted the model's predictions against the actual test data. The red dashed line represents the "Perfect Prediction" 45-degree line. The tight clustering of blue dots around this line proves the model's reliability.

![Actual vs Predicted](Screenshots/Actual%20vs%20Predicted%20Graph.png)
