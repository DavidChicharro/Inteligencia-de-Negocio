# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	Programa principal P2
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""


import pandas as pd

import kmeans as km
import meanshift as msh
import dbscan as dbs
import jerarquico as jer
import birch as bir

def norm_to_zero_one(df):
	return (df - df.min()) * 1.0 / (df.max() - df.min())

censo = pd.read_csv('./datos/mujeres_fecundidad_INE_2018.csv')

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
	  print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#censo = censo.replace(np.NaN,0)

#O imputar, por ejemplo con la media	  
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)


'''
Caso 1
'''

subset_caso1 = censo.loc[(censo['NDESEOHIJO']<8) & ((censo['EMPPAREJA']==1) | (censo['EMPPAREJA']==3))]
usadas1 = ['SATISTARDOM','SATISRELAC','NDESEOHIJO','ESTUDIOSA']
X1 = subset_caso1[usadas1]
X1_normal = X1.apply(norm_to_zero_one)
caso_estudio = 'Caso1'

#kmeans
km.kmeans(X1, X1_normal, usadas1, caso_estudio)

#MeanShift
msh.meanshift(X1, X1_normal, usadas1, caso_estudio)

#DBSCAN
dbs.dbscan(X1, X1_normal, usadas1, caso_estudio)

#Jerárquico
jer.jerarquico(X1, usadas1, caso_estudio,100)
jer.jerarquico(X1, usadas1, caso_estudio,4)

#Birch
#bir.birch(X1, X1_normal, usadas1, caso_estudio, 2) #no ejecuta


'''
Caso 2
'''

subset_caso2 = censo.loc[(censo['ANONPAR']>1950)]
usadas2 = ['TIPOUNION','EDAD','ANONPAR','ANOVIVJUN']
X2 = subset_caso2[usadas2]
X2_normal = X2.apply(norm_to_zero_one)
caso_estudio = 'Caso2'

#kmeans
km.kmeans(X2, X2_normal, usadas2, caso_estudio)

#MeanShift
msh.meanshift(X2, X2_normal, usadas2, caso_estudio)

#DBSCAN
dbs.dbscan(X2, X2_normal, usadas2, caso_estudio)

#Jerárquico
jer.jerarquico(X2, usadas2, caso_estudio,100)
jer.jerarquico(X2, usadas2, caso_estudio,5)

#Birch
bir.birch(X2, X2_normal, usadas2, caso_estudio, 2)


'''
Caso 3
'''

subset_caso3 = censo.loc[(censo['USOANTICONCEP']==1) & (censo['NDESEOHIJO']<8) & (censo['INGREHOG']<50000)]
usadas3 = ['EDAD','INGREHOG','NHIJOS','NDESEOHIJO']
X3 = subset_caso3[usadas3]
X3_normal = X3.apply(norm_to_zero_one)
caso_estudio = 'Caso3'

#kmeans
km.kmeans(X3, X3_normal, usadas3, caso_estudio)

#MeanShift
msh.meanshift(X3, X3_normal, usadas3, caso_estudio)

#DBSCAN
dbs.dbscan(X3, X3_normal, usadas3, caso_estudio) #0.2

#Jerárquico
jer.jerarquico(X3, usadas3, caso_estudio,100)
jer.jerarquico(X3, usadas3, caso_estudio,4)

#Birch
#bir.birch(X3, X3_normal, usadas3, caso_estudio, 2) #No ejecuta
