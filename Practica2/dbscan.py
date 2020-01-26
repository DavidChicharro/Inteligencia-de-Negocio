# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	DBSCAN
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""


import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from math import floor
import seaborn as sns

import plots as plts	
	

def dbscan(X, X_normal, usadas, caso_est):
	eps = float(input('Epsion: '))
	
	f = open('./{}/dbscan/resultados.txt'.format(caso_est), 'w')
	
	dbs = DBSCAN(eps=eps)
	
	print('----- Ejecutando dbscan', end='')	
	f.write('----- Ejecutando dbscan con eps={}'.format(eps))
	
	t = time.time()
	dbs.fit(X_normal)
	tiempo = time.time() - t
	
	labels = dbs.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)
	
	print(": {:.2f} segundos".format(tiempo), end='\n')
	f.write(': {:.2f} segundos \n'.format(tiempo))
	metrics_CH = metrics.calinski_harabasz_score(X_normal, labels)
	print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end='\n')
	f.write('Calinski-Harabaz Index: {:.3f}\n'.format(metrics_CH))
	
	#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
	if len(X) > 10000:
	   muestra_silhoutte = 0.2
	else:
	   muestra_silhoutte = 1.0
	   
	metric_SC = metrics.silhouette_score(X_normal, labels, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
	print("Silhouette Coefficient: {:.5f}".format(metric_SC), end='\n')
	f.write('Silhouette Coefficient: {:.5f}\n'.format(metric_SC))

	#se convierte la asignación de clusters a DataFrame
	clusters = pd.DataFrame(labels,index=X.index,columns=['cluster'])
	
	print('Epsilon: {}'.format(eps))
	print('Número estimado de clusters: {}'.format(n_clusters_))
	print('Número estimado de puntos ruidosos: {}'.format(n_noise_))
	
	print("Tamaño de cada cluster:")
	f.write('Tamaño de cada cluster:\n')
	size=clusters['cluster'].value_counts()
	for num,i in size.iteritems():
		print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
		f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
		
	f.close()

	X_DBSCAN = pd.concat([X_normal,clusters],axis=1)
	X_DBSCAN = X_DBSCAN[X_DBSCAN.cluster != -1]
	cluster_centers = X_DBSCAN.groupby('cluster').mean()
	
	centers = pd.DataFrame(cluster_centers,columns=list(X))
	plts.plot_graphics(X, usadas, centers, clusters, caso_est, 'dbscan')