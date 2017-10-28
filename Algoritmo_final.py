#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:42:18 2017

@author: veronica
"""
import numpy as np
from Parametros_Final import main1
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

#Adjuntos están los dos archivos que aquí se nombran

#fy = np.load('BPSK_Dataset_1024_500datos_minus20to18SNR_25datoscu.npy')
fy = np.load('BFSK_Dataset_1024_500datos_minus20to18SNR_25datoscu.npy')

#Se toma una fila de la matriz contenida en el archivo. En este caso
#se toma la fila 400. Los datos que ingresan a la clase son los de la FFT de la señal

ffty=abs(np.fft.fft(fy[400,1:]))
main1(ffty)     #Desde aquí se imprimen los valores de Amplitud, Frecuencia y Ancho de banda


mlp=joblib.load('Redes_Lin_3Layers_10Percep_cu.pkl') #Se carga el archivo que contiene el modelo de redes neuronales
R=mlp.predict(scaler.transform(fy[400,1:]))
if R==0:
    print 'Es BPSK'
else:
    print 'Es BFSK'
