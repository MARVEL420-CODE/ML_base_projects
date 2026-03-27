

# =========================
# 🚗 VEHICLE PRICE PREDICTION
# =========================

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =========================
# 1️⃣ LOAD DATASET
# =========================
file_path = r"C:\Users\marve\Desktop\dataset.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df.drop(columns=['name', 'description'], inplace=True, errors='ignore')

# Remove rows where price is missing
df.dropna(subset=['price'], inplace=True)

# Separate target variable
y = df['price']
X = df.drop(columns=['price'])

# Identify feature types
num_features = ['year', 'cylinders', 'mileage', 'doors']
cat_features = ['make', 'model', 'fuel', 'transmission', 'trim', 'body', 
                'exterior_color', 'interior_color', 'drivetrain']

# =========================
# 2️⃣ PREPROCESSING PIPELINE
# =========================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# =========================
# 3️⃣ TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4️⃣ RIDGE REGRESSION
# =========================
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge())
])

ridge_params = {'ridge__alpha': [0.1, 1, 10, 100]}

ridge_grid = GridSearchCV(
    ridge_pipeline, ridge_params, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
ridge_grid.fit(X_train, y_train)

# =========================
# 5️⃣ DECISION TREE REGRESSOR
# =========================
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeRegressor())
])

dt_params = {
    'dt__max_depth': [5, 10, 15, None],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 5]
}

dt_grid = GridSearchCV(
    dt_pipeline, dt_params, cv=5, scoring='r2', n_jobs=-1, verbose=1
)
dt_grid.fit(X_train, y_train)

# =========================
# 6️⃣ MODEL EVALUATION
# =========================
models = {'Ridge Regression': ridge_grid, 'Decision Tree': dt_grid}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Results:")
    print(f"✅ MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    
    # ✅ Fixed RMSE calculation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE: {rmse:.2f}")
    
    print(f"✅ R² Score: {r2_score(y_test, y_pred):.2f}\n")


# =========================
# 7️⃣ CROSS-VALIDATION SCORES
# =========================
ridge_cv_scores = cross_val_score(ridge_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')
dt_cv_scores = cross_val_score(dt_grid.best_estimator_, X_train, y_train, cv=5, scoring='r2')

plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), ridge_cv_scores, marker='o', label='Ridge Regression', color='blue')
plt.plot(range(1, 6), dt_cv_scores, marker='s', label='Decision Tree', color='red')
plt.axhline(y=np.mean(ridge_cv_scores), color='blue', linestyle='dashed', label='Ridge Avg Score')
plt.axhline(y=np.mean(dt_cv_scores), color='red', linestyle='dashed', label='DT Avg Score')
plt.xlabel("Cross-Validation Fold")
plt.ylabel("R² Score")
plt.title("Cross-Validation Results for Ridge & Decision Tree")
plt.legend()
plt.show()

# =========================
# 8️⃣ MANUAL PREDICTION FUNCTION
# =========================
def manual_prediction(model):
    print("\n⚡ Enter values for manual prediction (one by one):")
    user_input = {}

    for col in num_features:
        try:
            user_input[col] = float(input(f"Enter {col}: "))
        except ValueError:
            user_input[col] = np.nan  # Handle invalid numeric input

    for col in cat_features:
        user_input[col] = input(f"Enter {col}: ").strip()

    input_df = pd.DataFrame([user_input])

    # ✅ Directly predict (pipeline handles preprocessing)
    pred = model.predict(input_df)[0]
    print(f"\n🚗 Predicted Vehicle Price: ${pred:,.2f}")

# =========================
# 9️⃣ MODEL SELECTION + SAVE
# =========================
print("\n🛠️ Choose a model for prediction:")
choice = input("Type 'ridge' for Ridge Regression or 'tree' for Decision Tree: ").strip().lower()

if choice == 'ridge':
    best_model = ridge_grid.best_estimator_
    print("\n✅ You chose Ridge Regression model.")
    manual_prediction(best_model)
    joblib.dump(best_model, "ridge_model.pkl")
    print("💾 Ridge model saved as 'ridge_model.pkl'")

elif choice == 'tree':
    best_model = dt_grid.best_estimator_
    print("\n✅ You chose Decision Tree model.")
    manual_prediction(best_model)
    joblib.dump(best_model, "decision_tree_model.pkl")
    print("💾 Decision Tree model saved as 'decision_tree_model.pkl'")

else:
    print("❌ Invalid choice! Exiting.")
