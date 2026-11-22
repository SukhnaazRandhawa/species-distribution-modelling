"""
View the species dataset as tables.
Run from the species folder (with .venv activated):
    pip install pandas
    python view_data.py
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

# ---------- TRAINING DATA ----------
data = np.load("species_train.npz")
names = dict(zip(data["taxon_ids"], data["taxon_names"]))

train = pd.DataFrame({
    "latitude": data["train_locs"][:, 0],
    "longitude": data["train_locs"][:, 1],
    "taxon_id": data["train_ids"],
})
train["species_name"] = train["taxon_id"].map(names)

print("=== First 10 training sightings ===")
print(train.head(10), "\n")

print("=== Sightings per species (top 10) ===")
print(train["species_name"].value_counts().head(10), "\n")

print("=== Sightings per species (bottom 10) ===")
print(train["species_name"].value_counts().tail(10), "\n")

# Look at one species (change this name to any species you like)
example = train["species_name"].iloc[0]
print(f"=== All sightings of {example} (first 10) ===")
print(train[train["species_name"] == example].head(10), "\n")

# Save the full table so you can open it in Excel / Numbers
train.to_csv("train_table.csv", index=False)
print("Saved full training table to train_table.csv\n")

# ---------- TEST DATA ----------
test = np.load("species_test.npz", allow_pickle=True)
test_locs = test["test_locs"]
test_pos = dict(zip(test["taxon_ids"], test["test_pos_inds"]))

print("Number of test locations:", len(test_locs))

# Which species are present at one test location?
loc_index = 0
present = [names.get(sp, sp) for sp, inds in test_pos.items() if loc_index in set(inds)]
lat, lon = test_locs[loc_index]
print(f"\n=== Species present at test location {loc_index} "
      f"(lat {lat:.2f}, lon {lon:.2f}) ===")
print(f"{len(present)} species:", present[:20])