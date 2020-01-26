# -*- coding: utf-8 -*-
"""
Autor:
    David Carrasco Chicharro
Fecha:
    Diciembre/2019
Contenido:
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada

	Script 15 - Nepal Earthquake
"""

import pandas as pd

y1 = pd.read_csv('../submissions/submission_13.csv')
y2 = pd.read_csv('../submissions/submission_12.csv')
y3 = pd.read_csv('../submissions/submission_14.csv')
y4 = pd.read_csv('../submissions/submission_11.csv')
y5 = pd.read_csv('../submissions/submission_10.csv')

pred1 = y1['damage_grade'].values
pred2 = y2['damage_grade'].values
pred3 = y3['damage_grade'].values
pred4 = y4['damage_grade'].values
pred5 = y5['damage_grade'].values

total = len(pred1)
lista_preds = [pred1, pred2, pred3, pred4, pred5]
result = []

for i in range(total):
  valores = [0,0,0]  #veces que aparecen los valores 1,2,3
  for pred in lista_preds:
    if pred[i]==1:
      valores[0]+=1
    elif pred[i]==2:
      valores[1]+=1
    else:
      valores[2]+=1

  if 5 in valores:
    result.append(valores.index(5)+1)
  elif 4 in valores:
    result.append(valores.index(4)+1)
  elif 3 in valores:
    result.append(valores.index(3)+1)
  else:
    if valores[1] == 2:
        result.append(2)
    else:
        result.append(3)

df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = result
df_submission.to_csv("../submissions/submission_15.csv", index=False)
