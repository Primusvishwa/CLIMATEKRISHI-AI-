import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\scvst\Desktop\Pune Hackathon\ML_Dataset_Organic_v2_5926.csv")
print(f"✅ Loaded {len(df)} rows\n")

X = df[['Manure_rate', 'Compost_rate']]
y = df[['global_warming', 'freshwater_eutrophication', 'terrestrial_acidification', 'terrestrial_ecotoxicity']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaled versions for Ridge, Lasso, SVM
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

targets = ['Global Warming', 'Freshwater Eutrophication', 'Terrestrial Acidification', 'Terrestrial Ecotoxicity']

# ── Evaluation Helper ──────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)

    print(f"\n{'='*70}")
    print(f"  MODEL: {name}")
    print(f"{'='*70}")
    print(f"  {'Impact Category':<28} {'R²':>10} {'MAE':>14} {'RMSE':>14}")
    print(f"  {'-'*66}")

    r2_scores = []
    for i, t in enumerate(targets):
        r2   = r2_score(y_test.iloc[:, i], preds[:, i])
        mae  = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
        r2_scores.append(r2)
        print(f"  {t:<28} {r2:>10.6f} {mae:>14.6f} {rmse:>14.6f}")

    print(f"  {'-'*66}")
    print(f"  {'Average R²':<28} {np.mean(r2_scores):>10.6f}")
    return model

# ── Models ─────────────────────────────────────────────────────────────────────
print("\n🔬 Running model comparison — Organic System...\n")

# 1. Random Forest (no scaling needed)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate("Random Forest", rf, X_train, X_test)

# 2. Ridge Regression
ridge = MultiOutputRegressor(Ridge(alpha=1.0))
evaluate("Ridge Regression", ridge, X_train_sc, X_test_sc)

# 3. Lasso Regression
lasso = MultiOutputRegressor(Lasso(alpha=0.001))
evaluate("Lasso Regression", lasso, X_train_sc, X_test_sc)

# 4. SVM Regression
svm = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
evaluate("SVM Regression (RBF)", svm, X_train_sc, X_test_sc)

print(f"\n{'='*70}")
print("✅ Done. Compare Average R² across all models above.")
print(f"{'='*70}\n")