"""Merge raw/PCA rows with the corrected ceiling rows into one table."""

import pandas as pd

full = pd.read_csv("results_recover_full.csv")
full = full[~full.method.str.endswith("ceiling")]

ceil = pd.read_csv("results_ceiling.csv")
rows = []
for _, r in ceil.iterrows():
    rows.append({"pair": r["pair"],
                 "method": ("low_ceiling" if r["which"] == "M" else "high_ceiling"),
                 "target": r["target"], "other": r["other"]})

df = pd.concat([full, pd.DataFrame(rows)], ignore_index=True)
df.to_csv("results_recover_final.csv", index=False)

g = df.groupby("method")[["target", "other"]].mean()
g["acc"] = df.groupby("method").apply(lambda d: (d.target > d.other).mean())
print(g.round(3).to_string())
