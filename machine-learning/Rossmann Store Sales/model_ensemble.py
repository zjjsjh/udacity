import numpy as np
import pandas as pd

df_base = pd.read_csv("output/xgb_base.csv")
#df_best = pd.read_csv("output/test_predict/test_21005.csv")
df_best = pd.read_csv("output/test_predict/test_21005_final.csv")
df_worst = pd.read_csv("output/test_predict/test_23001.csv")
df_worst2 = pd.read_csv("output/test_predict/test_21001.csv")
df_worst.rename(columns={'Sales':'Sales3'}, inplace=True)
df_worst2.rename(columns={'Sales':'Sales4'}, inplace=True)
df_base.rename(columns={'Sales':'Sales1'}, inplace=True)
df_best.rename(columns={'Sales':'Sales2'}, inplace=True)

result = pd.merge(df_base, df_worst, on='ID')
result = pd.merge(result, df_best, on='ID')
result = pd.merge(result, df_worst2, on='ID')

alpha = 0.985
result['Sales'] = (result['Sales1'] + result['Sales2']
					+  result['Sales4'] ) / 3 * alpha
result.drop(['Sales1', 'Sales2', 'Sales3', 'Sales4'], axis=1, inplace=True)

result[['ID']] = result[['ID']].astype(int)
result.to_csv('output/ensemble/base_worst.csv', index=0)
#private score = 0.11837 alpha = 0.984 final/23001/21001/base
#private score = 0.11754 alpha = 0.985 final/21001/base
