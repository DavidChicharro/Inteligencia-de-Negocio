# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	k-means
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""


import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from math import floor
import seaborn as sns

import plots as plts

def ElbowMethod(X_normal, caso_est):
	wcss = []	# Within-Cluster Sums of Squares
	for i in range(1, 11):
		kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123456)
		kmeans.fit(X_normal)
		wcss.append(kmeans.inertia_)
	plt.plot(range(1, 11), wcss)
	plt.title('Elbow Method')
	plt.xlabel('Número de clusters')
	plt.ylabel('WCSS')
	plt.show()
	plt.savefig('./{}/kmeans/0_ElbowMethod.png'.format(caso_est), dpi=600)
	print("")
	

def kmeans(X, X_normal, usadas, caso_est):
	ElbowMethod(X_normal, caso_est)
	
	f = open('./{}/kmeans/resultados.txt'.format(caso_est), 'w')
	
	list_num_clus = list()
	for i in range (2,11):
		list_num_clus.append(i)
	
	for nc in list_num_clus:
		k_means = KMeans(init='k-means++', n_clusters=nc)
		
		print('----- Ejecutando kmeans para {} clusters'.format(nc),end='')	
		f.write('----- Ejecutando kmeans para {} clusters'.format(nc))
		
		t = time.time()   
		cluster_predict = k_means.fit_predict(X_normal)	
		tiempo = time.time() - t
		
		print(": {:.2f} segundos".format(tiempo), end='\n')
		f.write(': {:.2f} segundos \n'.format(tiempo))
		metrics_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
		print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end='\n')
		f.write('Calinski-Harabaz Index: {:.3f}\n'.format(metrics_CH))
		
		#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
		if len(X) > 10000:
		   muestra_silhoutte = 0.2
		else:
		   muestra_silhoutte = 1.0
		   
		metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
		print("Silhouette Coefficient: {:.5f}".format(metric_SC), end='\n')
		f.write('Silhouette Coefficient: {:.5f}\n'.format(metric_SC))
	
		#se convierte la asignación de clusters a DataFrame
		clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
		
		print("Tamaño de cada cluster:")
		f.write('Tamaño de cada cluster:\n')
		size=clusters['cluster'].value_counts()
		for num,i in size.iteritems():
			print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
			f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
		
	f.close()
	
	
	# Se selecciona el número de clusters con el que se va a trabajar
	num_clusters = int(input('Número de clusters: '))
	k_means = KMeans(init='k-means++', n_clusters=num_clusters)
	cluster_predict = k_means.fit_predict(X_normal)
	clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
	
	centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
	
	plts.plot_graphics(X, usadas, centers, clusters, caso_est, 'kmeans')