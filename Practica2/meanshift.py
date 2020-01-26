# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	MeanShift
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from math import floor
import seaborn as sns

import plots as plts

def meanshift(X, X_normal, usadas, caso_est):
	bw = estimate_bandwidth(X_normal, quantile=0.2, n_samples=500, random_state=123456)
	ms = MeanShift(bandwidth=bw, bin_seeding=True)
	
	f = open('./{}/meanshift/resultados.txt'.format(caso_est), 'w')
	
	print('----- Ejecutando MeanShift',end='')
	f.write('----- Ejecutando MeanShift')
	
	t = time.time()   
	ms.fit(X_normal)	
	tiempo = time.time() - t	
	
	labels = ms.labels_
	
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
	
	print("Tamaño de cada cluster:")
	f.write('Tamaño de cada cluster:\n')
	size=clusters['cluster'].value_counts()
	for num,i in size.iteritems():
		print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
		f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
			
	f.close()
	
	centers = pd.DataFrame(ms.cluster_centers_,columns=list(X))
	plts.plot_graphics(X, usadas, centers, clusters, caso_est, 'meanshift')
