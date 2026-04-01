import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\scvst\Desktop\Pune Hackathon\ML_Dataset_v2_7116.csv")
print(f"✅ Loaded {len(df)} rows\n")

X = df[['N_rate', 'P_rate', 'K_rate', 'Zn_rate']]
y = df[['global_warming', 'freshwater_eutrophication', 'terrestrial_acidification', 'terrestrial_ecotoxicity']]

# ── Train / Test Split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ── Train Ridge ────────────────────────────────────────────────────────────────
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train_sc, y_train)
preds = model.predict(X_test_sc)

# ── Evaluate ───────────────────────────────────────────────────────────────────
targets = ['Global Warming', 'Freshwater Eutrophication', 'Terrestrial Acidification', 'Terrestrial Ecotoxicity']

print(f"{'='*70}")
print(f"  MODEL: Ridge Regression — Conventional Rice")
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
print(f"{'='*70}\n")

# ── Save Model + Scaler ────────────────────────────────────────────────────────
joblib.dump(model,  r"C:\Users\scvst\Desktop\Pune Hackathon\model_conventional.pkl")
joblib.dump(scaler, r"C:\Users\scvst\Desktop\Pune Hackathon\scaler_conventional.pkl")

print("✅ Model saved  → model_conventional.pkl")
print("✅ Scaler saved → scaler_conventional.pkl")