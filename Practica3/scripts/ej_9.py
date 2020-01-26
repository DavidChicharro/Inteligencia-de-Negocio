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

	Script 9 - Nepal Earthquake
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import xgboost as xgb
from scipy import stats

data_x = pd.read_csv('../datos/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../datos/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../datos/nepal_earthquake_tst.csv')

# Mezcla en building_damage variable los datos con su clase correspondiente
building_damage = data_x.merge(data_y, how = 'inner', on = 'building_id')
building_damage['damage_grade'] = building_damage['damage_grade'].astype('category')

building_damage.loc[building_damage['age'] <=25 , 'age_group'] = 'New'
building_damage.loc[(building_damage['age'] > 25) & (building_damage['age'] <= 100),'age_group' ]= 'Old'
building_damage.loc[building_damage['age'] > 100 ,'age_group' ] = 'Very Old'
building_damage['age_group']= ""


# building_damage1 almacena los datos con su clase y se eliminan las
# columnas menos importantes
building_damage1 = data_x.merge(data_y, how = 'inner', on = 'building_id')
building_damage1 = building_damage1.drop(columns ="has_secondary_use")
building_damage1 = building_damage1.drop(columns ="has_secondary_use_agriculture")
building_damage = building_damage1[building_damage1['age'] <= 250]

cat_feats = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position','legal_ownership_status',
       'plan_configuration']

# Se convierten las variables categóricas en numéricas
train_final = pd.get_dummies(building_damage,columns=cat_feats,drop_first=True)
test_final = pd.get_dummies(data_x_tst,columns=cat_feats,drop_first=True)

y_train=train_final.damage_grade
train=train_final.drop('damage_grade',axis=1)
X = train_final.drop('damage_grade',axis=1)
y = train_final['damage_grade']


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


print("------ XGB...")
clf = xgb.XGBClassifier(max_depth=15, n_estimators = 1500, n_jobs=-1)
clf, y_test_clf = validacion_cruzada(clf,X.values,y.values,skf)

clf = clf.fit(X.values,y.values)
y_pred_tra = clf.predict(X.values)
print("F1 score (tra): {:.4f}".format(f1_score(y.values,y_pred_tra,average='micro')))

test_final.drop(labels=['has_secondary_use'], axis=1,inplace = True)
test_final.drop(labels=['has_secondary_use_agriculture'], axis=1,inplace = True)
y_pred_tst = clf.predict(test_final.values)

df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../submissions/submission_9.csv", index=False)
