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

	Script 7 - Nepal Earthquake
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('../datos/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../datos/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../datos/nepal_earthquake_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)
    
'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN

X = data_x_nan.values
X_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

from sklearn.metrics import f1_score

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (tst): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------

#'''
print("------ XGB...")
clf = xgb.XGBClassifier(max_depth=15, n_estimators = 1000,n_jobs=-1)
clf, y_test_clf = validacion_cruzada(clf,X,y,skf)

clf = clf.fit(X,y)
y_pred_tra_xgb = clf.predict(X)
print("F1 score (tra) XGB: {:.4f}".format(f1_score(y,y_pred_tra_xgb,average='micro')))
y_pred_tst_xgb = clf.predict(X_tst)
#'''

#'''
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='regression_l1',n_estimators=1000,n_jobs=-1)
lgbm, y_test_lgbm = validacion_cruzada(lgbm,X,y,skf)

lgbm = lgbm.fit(X,y)
y_pred_tra_lgbm = lgbm.predict(X)
print("F1 score (tra) LGBM: {:.4f}".format(f1_score(y,y_pred_tra_lgbm,average='micro')))
y_pred_tst_lgbm = lgbm.predict(X_tst)
#'''

print("------ Random Forest ...")
rf = RandomForestClassifier(n_estimators=1000, max_depth=15,n_jobs=-1)
rf, y_test_lgbm = validacion_cruzada(rf,X,y,skf)

rf = rf.fit(X,y)
y_pred_tra_rf = rf.predict(X)
print("F1 score (tra) RF: {:.4f}".format(f1_score(y,y_pred_tra_rf,average='micro')))
y_pred_tst_rf = rf.predict(X_tst)

total = len(X_tst)
lista_preds = [y_pred_tst_xgb, y_pred_tst_lgbm, y_pred_tst_rf]
result = []

for i in range(total):
  valores = [0,0,0]
  for pred in lista_preds:
    if pred[i]==1:
      valores[0]+=1
    elif pred[i]==2:
      valores[1]+=1
    else:
      valores[2]+=1

  if 3 in valores:
    result.append(valores.index(3)+1)
  elif 2 in valores:
    result.append(valores.index(2)+1)
  else:
    result.append(2)



df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = result
df_submission.to_csv("../submissions/submission_7.csv", index=False)
