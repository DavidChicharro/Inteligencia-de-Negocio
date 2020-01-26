# -*- coding: utf-8 -*-
"""
Autor:
	David Carrasco Chicharro
Fecha:
	Diciembre/2019
Contenido:
	Plots de distintas gráficas
	Inteligencia de Negocio
	Grado en Ingeniería Informática
	Universidad de Granada
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_heatmap(X, centers, caso_est, alg):
	centers_desnormal = centers.copy()
	
	#se convierten los centros a los rangos originales antes de normalizar
	for var in list(centers):
		centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
	
	print("---------- Preparando el heatmap...")
	axhm = plt.axes()
	hm = sns.heatmap(centers, cmap='YlGnBu', annot=centers_desnormal, fmt='.3f', ax=axhm)
	axhm.set_title('Heatmap')
	hm.set_ylim(len(centers),0)
	hm.figure.savefig('./{}/{}/heatmap.png'.format(caso_est, alg), dpi=600)
	

def plot_scatter_matrix(X_alg, caso_est, alg):
	print("---------- Preparando el scatter matrix...")
	sns.set()
	variables = list(X_alg)
	variables.remove('cluster')
	
	sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette='husl', plot_kws={"s": 25}, diag_kind="hist")
	sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
	sns_plot.savefig('./{}/{}/scatter_matrix.png'.format(caso_est, alg), dpi=600)
	print("")


def plot_kde(X_alg, k, n_var, usadas, caso_est, alg):
	print("---------- Preparando kde...")
	fig, axes = plt.subplots(k, n_var, sharey=True)	
	colors = sns.color_palette(palette=None, n_colors=k, desat=None)
	
	rango = []
	for j in range(n_var):
		rango.append([X_alg[usadas[j]].min(),X_alg[usadas[j]].max()])
	
	for i in range(k):
		dat_filt = X_alg.loc[X_alg['cluster']==i]
		for j in range(n_var):
			ax = sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
			ax.set_xlim(rango[j][0],rango[j][1])
			
	fig.savefig('./{}/{}/kde.png'.format(caso_est, alg), dpi=600)


def plot_boxplot(X_alg, k, n_var, usadas, caso_est, alg):
	print("---------- Preparando el box plot...")
	fig, axes = plt.subplots(k, n_var, sharey=True)
	colors = sns.color_palette(palette=None, n_colors=k, desat=None)
	
	rango = []
	for j in range(n_var):
		rango.append([X_alg[usadas[j]].min(),X_alg[usadas[j]].max()])
	
	for i in range(k):
		dat_filt = X_alg.loc[X_alg['cluster']==i]
		for j in range(n_var):
			ay = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i,j])
			ay.set_xlim(rango[j][0],rango[j][1])
			
	fig.savefig('./{}/{}/boxplot.png'.format(caso_est, alg), dpi=600)
	

def plot_graphics(X, usadas, centers, clusters, caso_est, alg):
	plot_heatmap(X, centers, caso_est, alg)
	
	X_alg = pd.concat([X, clusters], axis=1)
	plot_scatter_matrix(X_alg, caso_est, alg)
	
	size=clusters['cluster'].value_counts()
	k = len(size)
	n_var = len(usadas)
	plot_kde(X_alg, k, n_var, usadas, caso_est, alg)
	plot_boxplot(X_alg, k, n_var, usadas, caso_est, alg)
