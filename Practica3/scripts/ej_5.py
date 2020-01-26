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

	Script 5 - Nepal Earthquake
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
from sklearn.metrics import f1_score

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


'''
8 algs
'''
print("------ LightGBM 500...")
lgbm500 = lgb.LGBMClassifier(objective='regression_l1',n_estimators=500,n_jobs=-1)
clf_lgbm500 = lgbm500.fit(X,y)
y_pred_tra1 = clf_lgbm500.predict(X)

print("------ LightGBM 1000...")
lgbm1000 = lgb.LGBMClassifier(objective='regression_l1',n_estimators=1000,n_jobs=-1)
clf_lgbm1000 = lgbm1000.fit(X,y)
y_pred_tra2 = clf_lgbm1000.predict(X)

print("------ Random Forest 100...")
rf100_5 = RandomForestClassifier(n_estimators=100, max_depth=5,n_jobs=-1)
clf_rf100_5 = rf100_5.fit(X,y)
y_pred_tra3 = clf_rf100_5.predict(X)

print("------ Random Forest 500...")
rf500_5 = RandomForestClassifier(n_estimators=500, max_depth=5,n_jobs=-1)
clf_rf500_5 = rf500_5.fit(X,y)
y_pred_tra4 = clf_rf500_5.predict(X)

print("------ Random Forest 1000...")
rf1000_5 = RandomForestClassifier(n_estimators=1000, max_depth=5,n_jobs=-1)
clf_rf1000_5 = rf1000_5.fit(X,y)
y_pred_tra5 = clf_rf1000_5.predict(X)

print("------ Random Forest 100...")
rf100_10 = RandomForestClassifier(n_estimators=100, max_depth=10,n_jobs=-1)
clf_rf100_10 = rf100_10.fit(X,y)
y_pred_tra6 = clf_rf100_10.predict(X)

print("------ Random Forest 500...")
rf500_10 = RandomForestClassifier(n_estimators=500, max_depth=10,n_jobs=-1)
clf_rf500_10 = rf500_10.fit(X,y)
y_pred_tra7 = clf_rf500_10.predict(X)

print("------ Random Forest 1000...")
rf1000_10 = RandomForestClassifier(n_estimators=1000, max_depth=10,n_jobs=-1)
clf_rf1000_10 = rf1000_10.fit(X,y)
y_pred_tra8 = clf_rf1000_10.predict(X)

lista_predicciones = [y_pred_tra1,y_pred_tra2,y_pred_tra3,y_pred_tra4,y_pred_tra5,y_pred_tra6,y_pred_tra7,y_pred_tra8]
sumas = []
for i in range(0,len(y_pred_tra1)):
	suma = 0
	for pred in lista_predicciones:
		suma += abs(y[i]-pred[i])
	sumas.append(suma)

total = len(sumas)
mantener = []
for i in range(total):
    if sumas[i]<5:
        mantener.append(i)
        
X_nuevo = np.take(X,mantener,axis=0)
y_nuevo = np.take(y,mantener,axis=0)

#------------------------------------------------------------------------
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

#'''
print("------ XGB...")
clf = xgb.XGBClassifier(max_depth=15, n_estimators = 1000, n_jobs=-1)

clf = clf.fit(X_nuevo,y_nuevo)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../submissions/submission_5.csv", index=False)
