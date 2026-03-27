# ML_base_projects
# 🚗 Vehicle Price Prediction

## 📌 Project Overview

This project predicts the price of a vehicle based on various features such as year, mileage, fuel type, transmission, and more. It uses machine learning models to analyze data and provide accurate price predictions.

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn

---

## 🧠 Machine Learning Models

* Ridge Regression
* Decision Tree Regressor
* Hyperparameter tuning using GridSearchCV

---

## 🔄 Workflow

1. Data Loading & Cleaning
2. Feature Engineering
3. Data Preprocessing (Scaling + Encoding)
4. Model Training
5. Model Evaluation (MAE, RMSE, R² Score)
6. Cross-validation
7. Manual Prediction System
8. Model Saving using Joblib

---

## 📊 Features Used

* Numerical:

  * Year
  * Mileage
  * Cylinders
  * Doors

* Categorical:

  * Make
  * Model
  * Fuel Type
  * Transmission
  * Body Type
  * Colors
  * Drivetrain

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-price-prediction.git
cd vehicle-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script

```bash
python vehicle_price_prediction.py
```

---

## ⚠️ Important Note

Make sure the dataset file (`dataset.csv`) is in the same folder as the script.

---

## 💾 Output

* Displays model performance metrics:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

* Allows manual input for prediction

* Saves trained model as:

  * `ridge_model.pkl` OR
  * `decision_tree_model.pkl`

---

## 🚀 Future Improvements

* Add more advanced models (Random Forest, XGBoost)
* Build a web app using Streamlit
* Improve dataset quality and size
* Deploy the model online

---

## 👨‍💻 Author

**Purushoth Marvel**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
