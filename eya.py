import pandas as pd
df = pd.read_excel("BCT_RAGAS_Test_Table_FILLED.xlsx", sheet_name="RAGAS_Test_Set")

# Ne garder que les lignes réellement traitées (answer non vide)
traitees = df[df["Réponse générée (answer)"].fillna("").str.strip() != ""]

print(f"{len(traitees)} questions traitées")
for _, row in traitees.iterrows():
    print(f"\nQ{row['N°']}: {row['Question']}")
    print(f"→ {row['Réponse générée (answer)']}")