"""
Construit TP1/dataset.csv à partir du dataset officiel d'adoption des bus
scolaires électriques (World Resources Institute / Electric School Bus Initiative).

Pré-requis : télécharger et décompresser le dataset v8 dans _original_dataset/
  Source : https://datasets.wri.org/dataset/electric-school-bus-adoption
  (feuille utilisée : "1. District-level data")

Étapes :
  - sélection minimale ciblée (9 features d'origine + 3 à fort signal)
  - exclusion des variables de fuite (bloc 3*, modalités "Unknown")
  - encodage cible yes/no -> 1/0, imputation des NA
"""
import pandas as pd
import numpy as np
import os

base = "_original_dataset/electric_school_bus_adoption_dataset_v8_2024-08Aug"
f = os.path.join(base, "ESB_adoption_dataset_v8_update_august_2024.xlsx")
df = pd.read_excel(f, sheet_name="1. District-level data")

target = "0a. Has committed ESBs?"

# 9 features d'origine + 3 ajoutees (minimal cible)
categorical_features = [
    "1p. Locale broad type (name)",
    "1q. Census Region",
    "5o. EPA 2022 Clean School Bus Rebate Program prioritized school district?",  # AJOUT (politique publique)
]
numeric_features = [
    "2a. Total number of buses",
    "4b. Number of students in district",
    "4c. Number of schools in district",
    "4f. Median household income",
    "4g. Percent of population below the poverty level",
    "5f. PM2.5 concentration",
    "5h. Ozone concentration",
    "5b. Percent non-white and/or Hispanic",  # AJOUT (equite)
]
keep = categorical_features + numeric_features + [target]

out = df[keep].copy()

# --- Cible: yes/no -> 1/0 ---
out[target] = (out[target].astype(str).str.lower() == "yes").astype(float)

# --- Neutraliser les pieges (Unknown = marqueur indirect de la cible) ---
out["1p. Locale broad type (name)"] = out["1p. Locale broad type (name)"].replace("Unknown", np.nan)

# --- Normaliser la casse de 5o (il y a 'Yes' et 'yes') ---
col5o = "5o. EPA 2022 Clean School Bus Rebate Program prioritized school district?"
out[col5o] = out[col5o].astype(str).str.strip().str.capitalize()  # Yes / No
out[col5o] = out[col5o].replace({"Nan": np.nan})

# --- Imputation ---
# categorielles: mode ; numeriques: mediane
for c in categorical_features:
    out[c] = out[c].fillna(out[c].mode(dropna=True)[0])
for c in numeric_features:
    out[c] = out[c].fillna(out[c].median())

print("AVANT imputation NA total cible:", df[target].isna().sum())
print("SHAPE final:", out.shape)
print("NA restants:", out.isna().sum().sum())
print("\nCible:")
print(out[target].value_counts())
print("ratio classe 1:", round(out[target].mean()*100, 2), "%")
print("\nColonnes:", list(out.columns))
print("\n5o distribution:")
print(out[col5o].value_counts())

out.to_csv("TP1/dataset.csv", index=False)
print("\n=> TP1/dataset.csv ECRIT")
