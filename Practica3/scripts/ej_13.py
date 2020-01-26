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

	Script 13 - Nepal Earthquake
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


# Rango intercuartílico
# División entre entidades numéricas y categórcias
building_damage_num_train = building_damage.select_dtypes(include=["number"])
building_damage_cat_train = building_damage.select_dtypes(exclude=["number"])

# Binarización entre V/F para los valores numéricos en el intervalo 3*sigma de una distribución normal
idx = np.all(stats.zscore(building_damage_num_train) < 3, axis=1)
Q1 = building_damage_num_train.quantile(0.02)
Q3 = building_damage_num_train.quantile(0.98)
IQR = Q3 - Q1
idx = ~((building_damage_num_train < (Q1 - 1.5 * IQR)) | (building_damage_num_train > (Q3 + 1.5 * IQR))).any(axis=1)
building_damage_cleaned = pd.concat([building_damage_num_train.loc[idx], building_damage_cat_train.loc[idx]], axis=1)


'''Selección de características'''
# Quedan fuera 'plan_configuration' y 'legal_ownership_status'
cat_feats = ['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position']

# Se convierten las variables categóricas en numéricas
train_final = pd.get_dummies(building_damage_cleaned,columns=cat_feats,drop_first=True)
test_final = pd.get_dummies(data_x_tst,columns=cat_feats,drop_first=True)

#from sklearn.model_selection import train_test_split
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


X = X.select_dtypes(exclude=['object'])

print("------ XGB...")
clf = xgb.XGBClassifier(max_depth=10, n_estimators = 1000, learning_rate=0.05, n_jobs=-1)
clf, y_test_clf = validacion_cruzada(clf,X.values,y.values,skf)


clf = clf.fit(X.values,y.values)
y_pred_tra = clf.predict(X.values)
print("F1 score (tra): {:.4f}".format(f1_score(y.values,y_pred_tra,average='micro')))


test_final = pd.get_dummies(data_x_tst,columns=cat_feats,drop_first=True)
test_final.drop(labels=['plan_configuration'], axis=1,inplace = True)
test_final.drop(labels=['legal_ownership_status'], axis=1,inplace = True)


y_pred_tst = clf.predict(test_final.values)

df_submission = pd.read_csv('../datos/nepal_earthquake_submission_format.csv')
df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../submissions/submission_13.csv", index=False)
