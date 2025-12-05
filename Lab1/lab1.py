import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

df = pd.read_csv("/Users/timataskin/Desktop/AI_ML/Lab1/SkillCraft.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)

print(f"\nCorrelation:\n{df.corr()}")

# Удаляем строки с NaN 
df = df.dropna()
print("\nAfter dropna size:", df.shape)

# Приводим строки, похожие на числа, к numeric
for col in df.columns:
    if df[col].dtype == 'object':
        s = df[col].dropna().astype(str).str.strip()
        if len(s) == 0:
            continue
        # если >70% строк выглядят как число, приводим к числовому
        num_like = s.str.match(r'^-?\d+(\.\d+)?$').sum()
        if num_like / len(s) > 0.7:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.replace(' ', ''), errors='coerce')

# ещё раз удалим возможные NaN, образовавшиеся при приведении 
df = df.dropna()
print("After type conversion size:", df.shape)
print("Dtypes now:\n", df.dtypes)

# Сохраняем текущее состояние обработанной таблицы
out_dir = "C:/Users/taski/Desktop/AI_ML/Lab1"
os.makedirs(out_dir, exist_ok=True)
processed_path = os.path.join(out_dir, "SkillCraft_processed.csv")
df.to_csv(processed_path, index=False)
print("Saved processed CSV:", processed_path)

# Корреляция числовых признаков
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
print("\nCorrelation matrix (numeric):\n", corr.round(3))
corr.to_csv(os.path.join(out_dir, "correlation_matrix.csv"))

# Выбор target и признаков
if "APM" not in df.columns:
    raise SystemExit("Нет колонки APM в таблице. Проверь файл.")

y = df["APM"]

# x1 — одна переменная
x1 = df[["HoursPerWeek"]] if "HoursPerWeek" in df.columns else df[[df.columns[0]]]

# Набор 1
x_set1_cols = [c for c in ("HoursPerWeek", "TotalHours", "ActionLatency") if c in df.columns]
x_set1 = df[x_set1_cols]

# Набор 2
x_set2_cols = [c for c in ("WorkersMade", "UniqueUnitsMade", "ComplexUnitsMade") if c in df.columns]
x_set2 = df[x_set2_cols]

print("\nSelected features:")
print("x1:", x1.columns.tolist())
print("x_set1:", x_set1_cols)
print("x_set2:", x_set2_cols)

# Удаление выбросов — простая фильтрация 1%-99% по используемым колонкам
used_cols = set(x1.columns.tolist() + x_set1_cols + x_set2_cols + ["APM"])
used_cols = [c for c in used_cols if c in df.columns]
df_no_out = df.copy()
for c in used_cols:
    if np.issubdtype(df_no_out[c].dtype, np.number):
        q_low = df_no_out[c].quantile(0.01)
        q_high = df_no_out[c].quantile(0.99)
        df_no_out = df_no_out[(df_no_out[c] >= q_low) & (df_no_out[c] <= q_high)]

print("Size before/after outlier removal:", df.shape, "->", df_no_out.shape)
df_no_out.to_csv(os.path.join(out_dir, "SkillCraft_no_outliers.csv"), index=False)

# Формируем train/test
def prepare_xy(df_use, feature_cols, target_col="APM"):
    X = df_use[feature_cols].astype(float)
    y_local = df_use[target_col].astype(float)
    mask = X.notnull().all(axis=1) & y_local.notnull()
    return X[mask], y_local[mask]

X1, Y1 = prepare_xy(df_no_out, x1.columns.tolist())
X2, Y2 = prepare_xy(df_no_out, x_set1_cols) if x_set1_cols else (pd.DataFrame(), pd.Series())
X3, Y3 = prepare_xy(df_no_out, x_set2_cols) if x_set2_cols else (pd.DataFrame(), pd.Series())

# split (test 20%)
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, Y1, test_size=0.2, random_state=0)
if not X2.empty:
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, Y2, test_size=0.2, random_state=0)
else:
    X2_tr = X2_te = y2_tr = y2_te = None
if not X3.empty:
    X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, Y3, test_size=0.2, random_state=0)
else:
    X3_tr = X3_te = y3_tr = y3_te = None

print("\nTrain/test sizes:")
print("m1:", X1_tr.shape, X1_te.shape)
if X2_tr is not None:
    print("m2:", X2_tr.shape, X2_te.shape)
if X3_tr is not None:
    print("m3:", X3_tr.shape, X3_te.shape)

# Построение моделей
model1 = LinearRegression()
model1.fit(X1_tr, y1_tr)

model2 = None
if X2_tr is not None:
    model2 = LinearRegression()
    model2.fit(X2_tr, y2_tr)

model3 = None
if X3_tr is not None:
    model3 = LinearRegression()
    model3.fit(X3_tr, y3_tr)

# Анализ моделей: коэффициенты, R2, RMSE
def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    return pred, r2, rmse

pred1, r2_1, rmse_1 = evaluate(model1, X1_te, y1_te)

print("\nm1 (one feature):")
print(" Coef:", model1.coef_, "Intercept:", model1.intercept_)
print(f" R2: {r2_1:.4f}  RMSE: {rmse_1:.4f}")

if model2 is not None:
    pred2, r2_2, rmse_2 = evaluate(model2, X2_te, y2_te)
    print("\nm2 (set1):")
    print(" Coefs:", model2.coef_, "Intercept:", model2.intercept_)
    print(f" R2: {r2_2:.4f}  RMSE: {rmse_2:.4f}")
else:
    pred2 = None

if model3 is not None:
    pred3, r2_3, rmse_3 = evaluate(model3, X3_te, y3_te)
    print("\nm3 (set2):")
    print(" Coefs:", model3.coef_, "Intercept:", model3.intercept_)
    print(f" R2: {r2_3:.4f}  RMSE: {rmse_3:.4f}")
else:
    pred3 = None

# График прямой поверх точек (для m1)
plt.figure(figsize=(6,4))
plt.scatter(X1_te.values.flatten(), y1_te.values, alpha=0.6)
xs = np.linspace(X1_te.values.min(), X1_te.values.max(), 100)
k = model1.coef_[0]
b = model1.intercept_
plt.plot(xs, k*xs + b)
plt.xlabel(X1.columns[0])
plt.ylabel("APM")
plt.title("m1: scatter + line")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "m1_scatter_line.png"))
plt.close()
print("Saved m1_scatter_line.png")

# Pred vs Actual (берём m2 если есть, иначе m1)
if pred2 is not None:
    plt.figure(figsize=(5,5))
    plt.scatter(y2_te, pred2, alpha=0.6)
    mn, mx = min(y2_te.min(), pred2.min()), max(y2_te.max(), pred2.max())
    plt.plot([mn,mx],[mn,mx], linestyle='--')
    plt.xlabel("Y_test"); plt.ylabel("Y_pred"); plt.title("m2: Y_pred vs Y_test")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "m2_pred_vs_actual.png"))
    plt.close()
    print("Saved m2_pred_vs_actual.png")
else:
    plt.figure(figsize=(5,5))
    plt.scatter(y1_te, pred1, alpha=0.6)
    mn, mx = min(y1_te.min(), pred1.min()), max(y1_te.max(), pred1.max())
    plt.plot([mn,mx],[mn,mx], linestyle='--')
    plt.xlabel("Y_test"); plt.ylabel("Y_pred"); plt.title("m1: Y_pred vs Y_test")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "m1_pred_vs_actual.png"))
    plt.close()
    print("Saved m1_pred_vs_actual.png")

# Гистограмма остатков
if pred2 is not None:
    residuals = pred2 - y2_te
    fname = "m2_residuals_hist.png"
else:
    residuals = pred1 - y1_te
    fname = "m1_residuals_hist.png"

plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30)
plt.title("Residuals histogram")
plt.xlabel("Y_pred - Y_test")
plt.savefig(os.path.join(out_dir, fname))
plt.close()
print("Saved", fname)

# Короткая итоговая сводка
print("\nSummary:")
print("Initial size:", pd.read_csv("C:/Users/taski/Desktop/AI_ML/Lab1/SkillCraft.csv").shape)
print("After cleaning:", df.shape)
print("After outlier removal:", df_no_out.shape)
print("Target: APM")
print("m1 feature:", x1.columns.tolist())
print("m2 features:", x_set1_cols)
print("m3 features:", x_set2_cols)
print("R2 (m1):", round(r2_1,4), "RMSE (m1):", round(rmse_1,4))
print("Files (processed CSV, no_outliers, correlation, plots) saved to:", out_dir)
