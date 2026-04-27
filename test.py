import pandas as pd
import statsmodels.api as sm

# =========================
# 1️⃣ Data
# =========================
data = pd.DataFrame({
    "x1": [-1, -1, 0, 1, 1],
    "x2": [-1, 0, 0, 0, 1],
    "y":  [7.2, 8.1, 9.8, 12.3, 12.9]
})

# =========================
# 2️⃣ Fit model
# =========================
X = sm.add_constant(data[["x1", "x2"]])
y = data["y"]
model = sm.OLS(y, X).fit()

# =========================
# 3️⃣ Use statsmodels influence functions
# =========================
influence = model.get_influence()

# Studentized (internally studentized) residuals
studentized_resid = influence.resid_studentized_internal

# Cook’s distance
cooks_d, pvals = influence.cooks_distance

# Leverage (hat diagonals)
hat_diag = influence.hat_matrix_diag

# Combine all results
results = pd.DataFrame({
    "Obs": range(1, len(y) + 1),
    "Residual": model.resid.round(2),
    "Hat_ii": hat_diag.round(1),
    "Studentized_r": studentized_resid.round(3),
    "Cook_D": cooks_d.round(4)
})

print(results)
