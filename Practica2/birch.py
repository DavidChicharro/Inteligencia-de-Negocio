# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	BIRCH
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""


import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import Birch
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from math import floor
import seaborn as sns

import plots as plts
	

def birch(X, X_normal, usadas, caso_est, num_clusters):	
	f = open('./{}/birch/resultados.txt'.format(caso_est), 'w')
	
	b = Birch(n_clusters = num_clusters)
	
	print('----- Ejecutando birch con {} clusters'.format(num_clusters), end='')	
	f.write('----- Ejecutando birch con {} clusters'.format(num_clusters))
	
	t = time.time()
	b.fit(X_normal)
	tiempo = time.time() - t
	
	labels = b.labels_
	
	print(": {:.2f} segundos".format(tiempo), end='\n')
	f.write(': {:.2f} segundos \n'.format(tiempo))
	
	try:
		metrics_CH = metrics.calinski_harabasz_score(X_normal, labels)
		print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end='\n')
		f.write('Calinski-Harabaz Index: {:.3f}\n'.format(metrics_CH))
	except ValueError:
		print('Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)')

	
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
	
	print("Tamaño de cada cluster:")
	f.write('Tamaño de cada cluster:\n')
	size=clusters['cluster'].value_counts()
	for num,i in size.iteritems():
		print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
		f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
		
	f.close()

	X_birch = pd.concat([X_normal,clusters],axis=1)
	cluster_centers = X_birch.groupby('cluster').mean()
	
	centers = pd.DataFrame(cluster_centers,columns=list(X))
	plts.plot_graphics(X, usadas, centers, clusters, caso_est, 'birch')