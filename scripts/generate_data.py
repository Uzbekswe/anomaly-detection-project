"""Generate synthetic CMAPSS-like data for testing the full pipeline."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Generate train_FD001.txt: 100 engines, variable cycle lengths
rows = []
for unit_id in range(1, 101):
    max_cycle = np.random.randint(128, 362)
    for cycle in range(1, max_cycle + 1):
        op1 = np.random.uniform(-0.0089, 0.0087)
        op2 = np.random.uniform(-0.0004, 0.0007)
        op3 = round(float(np.random.choice([100, 60, 42, 25, 20, 10, 0])), 2)
        decay = max(0, 1 - (max_cycle - cycle) / max_cycle * 0.3)
        sensors = []
        for s in range(1, 22):
            base = {
                1: 518.67, 2: 642.15, 3: 1589.70, 4: 1400.60, 5: 14.62,
                6: 21.61, 7: 554.36, 8: 2388.02, 9: 9046.19, 10: 1.30,
                11: 47.47, 12: 521.66, 13: 2388.02, 14: 8138.62, 15: 8.4195,
                16: 0.03, 17: 392, 18: 2388, 19: 100.00, 20: 39.06, 21: 23.4190,
            }.get(s, 500)
            noise = np.random.randn() * base * 0.002
            trend = (
                base * (decay - 1) * 0.05
                * (1 if s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21] else 0)
            )
            sensors.append(round(base + noise + trend, 4))
        rows.append([unit_id, cycle, op1, op2, op3] + sensors)

cols = (
    ["unit_id", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
df_train = pd.DataFrame(rows, columns=cols)
df_train.to_csv(raw_dir / "train_FD001.txt", sep=" ", index=False, header=False)
print(f"train_FD001.txt: {len(df_train)} rows, {df_train['unit_id'].nunique()} engines")

# Generate test_FD001.txt: truncated versions
test_rows = []
for unit_id in range(1, 101):
    unit_data = df_train[df_train["unit_id"] == unit_id]
    cutoff = max(30, len(unit_data) - np.random.randint(10, 50))
    test_rows.append(unit_data.iloc[:cutoff])
df_test = pd.concat(test_rows, ignore_index=True)
df_test.to_csv(raw_dir / "test_FD001.txt", sep=" ", index=False, header=False)
print(f"test_FD001.txt: {len(df_test)} rows")

# Generate RUL_FD001.txt
rul_values = []
for unit_id in range(1, 101):
    total = len(df_train[df_train["unit_id"] == unit_id])
    used = len(df_test[df_test["unit_id"] == unit_id])
    rul_values.append(total - used)
pd.DataFrame(rul_values).to_csv(
    raw_dir / "RUL_FD001.txt", sep=" ", index=False, header=False
)
print(f"RUL_FD001.txt: {len(rul_values)} engines")
print("Done! All 3 CMAPSS FD001 files generated.")
