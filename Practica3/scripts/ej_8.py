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

	Script 8 - Nepal Earthquake
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from scipy import stats

data_x = pd.read_csv('../datos/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../datos/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../datos/nepal_earthquake_tst.csv')

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


X = X.select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestClassifier
print("------ Random Forest...")
clf = RandomForestClassifier(max_depth=18, n_estimators = 2500, n_jobs=-1)
clf, y_test_clf = validacion_cruzada(clf,X.values,y.values,skf)

clf = clf.fit(X.values,y.values)
y_pred_tra = clf.predict(X.values)
print("F1 score (tra): {:.4f}".format(f1_score(y.values,y_pred_tra,average='micro')))

y_pred_tst = clf.predict(test_final.values)

df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../submissions/submission_8.csv", index=False)
