import pandas as pd

df = pd.read_csv("data/knowledge/drugs.csv")

print(df["tty"].value_counts())