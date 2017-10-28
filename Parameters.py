#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 1 11:56:24 2017

@author: veronica
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from Amplitude_Envelope import Amplitude_Envelope
import random

class Parameters():

    def __init__(self,entry,ffty):
        _ffty=ffty
        self.entry=entry

        #Se toman solo las frecuencias positivas de la FFT
        _fyc=np.zeros(int(len(_ffty)/2.0))
        if entry==1:
            _fc=abs(_ffty[:int(len(_ffty)/2)])
            #Para simular un offset en la frecuencia de las señales BPSK
            #se hizo un shift a la FFT y se centró en un valor aleatorio
            PSK=1
            if PSK==1:
                idx=int(len(_fc)/2.0)
                random.seed(21)
                idx=random.randrange(100, 400, 1)
                _fyc[:idx]=_fc[-idx:]
                _fyc[idx:]=_fc[:-idx]
                plt.plot(_fyc, color='r' ,label=u'Señal')
            else:
                _fyc=_fc
        elif entry==2:
            _fyc=_ffty

        AE=Amplitude_Envelope(_fyc,20)
        self.fty=AE.Amp_env()  #Se hace la aproximación por picos
#        AE.Graphic()   #Se grafica la aproximación por picos
        _Fs=200000      #Esta es la frecuencia de muestreo con que se generaron los datos
        _Fs=float(_Fs)
        self.L=float(len(self.fty))
        self.Lint=len(self.fty)
        self.freq=np.arange(0,_Fs,_Fs/(self.L),dtype='float')
        self.freq_re=np.arange(0,_Fs/2.0,_Fs/(2.0*self.L),dtype='float')

        self.off=self.fty.argmax()  #El offset inicial es el punto en donde se encuentra el máximo de la FFT
        self.offset=self.freq[self.fty.argmax()]
        self.amp=np.max(self.fty)   #La amplitud inicial es el valor máximo de la FFT
        self.width=0.1      #Este parámetro se fijó al observar que funcionaba bien con los sets trabajados
        self.alpha=self.width/self.L
        self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset))) #Definición de la aprox. inicial
        self.off=self.sinc.argmax()
        _gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
        _gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
        self.MainLobe=np.arange(_gm,self.off+_gp,1)  #Coordenada X de los puntos contenidos en el lóbulo principal


    def Amplitud(self):
        gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
        gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
        self.MainLobe=np.arange(gm,self.off+gp,1)
        j=0
        ValAmp=np.array([0,6],dtype='float')
        landa=0.5
        #El criterio de parada se definió por un error cuadrático medio
        Err=100
        while Err>=10:
            self.sinc=abs(np.sinc(self.alpha*(self.freq-self.offset)))
            V=abs(self.fty-self.sinc)
            num=0
            den=0
            for i in self.MainLobe:
                num=num+V[i]*self.sinc[i]/self.amp
                den=den+(self.sinc[i]/self.amp)**2
            deltaAmp=num/den
            self.amp=(1-landa)*self.amp+landa*deltaAmp
            j+=1
            ValAmp=np.resize(ValAmp,j+1)
            ValAmp[j]=self.amp
            Err=(mse(self.fty[self.MainLobe[0]:self.MainLobe[-1]],self.sinc[self.MainLobe[0]:self.MainLobe[-1]]))/self.amp
        self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset)))

    def Offset(self):
        j=0
        deltaOffset=np.zeros(2)
        while deltaOffset[-1]*deltaOffset[-2]>=0:
            gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
            gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
            self.MainLobe=np.arange(gm,self.off+gp,1)
            Lmain=len(self.MainLobe)
            V=abs(self.fty-self.sinc)
            cl=np.min([np.max(V[self.MainLobe[0]:self.off]),np.max(self.sinc[self.MainLobe[0]:self.off])])
            ML=0
            for i in range(self.MainLobe[0],self.off+1):
                ML=ML+np.clip(V[i],0,cl)-np.clip(self.sinc[i],0,cl)
            ML=ML/Lmain
            cl=np.min([np.max(V[self.off:self.MainLobe[-1]]),np.max(self.sinc[self.off:self.MainLobe[-1]])])
            MR=0
            for i in range(self.off,self.MainLobe[-1]+1):
                MR=MR+np.clip(V[i],0,cl)-np.clip(self.sinc[i],0,cl)
            MR=MR/Lmain
            if ML-MR > 0:
                deltaOffset[j]=1
            else:
                deltaOffset[j]=-1
            self.offset=self.offset+1*deltaOffset[j]
            self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset)))
            self.off=self.sinc.argmax()
            j+=1
            deltaOffset=np.resize(deltaOffset,j+1)

    def Width(self):
        beta=1
        j=0
        #El criterio de parada se definió por un error cuadrático medio
        Err=1
        if self.entry==1:
            ref=0.004
        elif self.entry==2:
            ref=0.005
        while Err>=ref:
            gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
            gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
            self.MainLobe=np.arange(gm,self.off+gp,1)
            V=np.clip(abs(self.fty-self.sinc),0,self.amp)
            num=0.0
            den=0.0
            for i in self.MainLobe:
                num=num+(self.amp-V[i])*((i-self.off)**2)
                den=den+(self.amp/(self.L**2))*((i-self.off)**4)
            Ev=np.sqrt(num/den)
            i=self.MainLobe[0]
            num=0
            den=0
            while i <= self.off:
                gamma=(self.L**2)*((self.amp-V[i])/self.amp)
                num=num+gamma
                den=den+(i-self.off)*(-gamma)
                i+=1
            i-=1
            while i <= self.MainLobe[-1]:
                gamma=(self.L**2)*((self.amp-V[i])/self.amp)
                num=num+gamma
                den=den+(i-self.off)*gamma
                i+=1
            Eh=num/den
            Eh=0
            deltaWidth=(1-beta)*Ev+beta*Eh
            self.alpha=0.9487*self.alpha-1*deltaWidth
            self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset)))
            Err=(mse(self.fty[self.MainLobe[0]:self.MainLobe[-1]],self.sinc[self.MainLobe[0]:self.MainLobe[-1]]))/self.amp

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def main(ffty,Parametros_cls=Parametros, options=None):
#def main1(Parametros_cls=Parametros, options=None):

    #Adjuntos están los dos archivos que aquí se nombran
#    fy = np.load('BPSK_Dataset_1024_500datos_minus20to18SNR_25datoscu.npy')
#    fy = np.load('BFSK_Dataset_1024_500datos_minus20to18SNR_25datoscu.npy')

    #Se toma una fila de la matriz contenida en el archivo. En este caso
    #se toma la fila 400. Los datos que ingresan a la clase son los de la FFT de la señal
#    ffty=abs(np.fft.fft(fy[400,1:]))

    #entry=1 corresponde a la primera aproximación con el algoritmo completo
    entry=1
    Param = Parametros_cls(entry,ffty)  #Inicializa la clase
    #Se define un error y se empieza a iterar con cada parámetro
    Err=100000.0
    while Err>0.05:
        Param.Amplitud()
        Param.Offset()
        Param.Width()
        Err=(mse(Param.fty[Param.MainLobe[0]:Param.MainLobe[-1]],Param.sinc[Param.MainLobe[0]:Param.MainLobe[-1]]))/Param.amp
    #Se calcula el error resultante para determinar si es necesario realizar
    #otra aproximación
    ErrTodo=(mse(Param.fty,Param.sinc))/Param.amp

###############################################################################
#                               Amplitud
#   (Aquí se hace uso de la caracterización de la potencia de recepción)
###############################################################################
    Carac=np.load('Rx_Power_Characterization_Gain_0_Regression.npy')
    if ((Param.offset/2.0)>=144e6) and ((Param.offset/2.0)<=148e6):
        idx=50e3    #Paso en frecuencias
        x1=Carac[int(2e6/idx):int(2e6/idx)+20,0]
        y1=Carac[int(2e6/idx):int(2e6/idx)+20,1]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    elif ((Param.offset/2.0)>=430e6) and ((Param.offset/2.0)<=440e6):
        #Para UHF
        fact=1600
        idx=50e3    #Paso en frecuencias
        x1=Carac[fact+int(5e6/idx):fact+int(5e6/idx)+20]
        y1=Carac[fact+int(5e6/idx):fact+int(5e6/idx)+20]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    elif ((Param.offset/2.0)>=2.4e9) and ((Param.offset/2.0)<=2.5e9):
        #Para Banda S
        fact=5600
        idx=1e6     #Paso en frecuencias
        x1=Carac[fact+int(500e6/idx):fact+int(500e6/idx)+20]
        y1=Carac[fact+int(500e6/idx):fact+int(500e6/idx)+20]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    else:
        Amp=Param.amp
###############################################################################


    sinc=Param.sinc
    amp=Param.amp
    freq=Param.offset/2.0
    band=Param.freq_re[Param.MainLobe[-1]]-Param.freq_re[Param.MainLobe[0]]

    print 'Amplitude: ', Amp
    print 'Frequency: ', freq
    print 'Bandwidth: ', band

    plt.plot(sinc,color='g',label=u'Aproximación final')
    plt.plot(Param.fty,color='k',label=u'Aproximación por picos')

    #Si el error resultante supera 0.007, se hace otra aproximación
    if ErrTodo >= 0.007:
        entry=2
        #La función que ingresa en la segunda aproximación es la señal residuo
        fy=abs(Param.fty-sinc)
        Param = Parametros_cls(entry,fy)
        Err=100000.0
        while Err>0.05:
            Param.Amplitud()
            Param.Offset()
            Param.Width()
            Err=mse(Param.fty[Param.MainLobe[0]:Param.MainLobe[-1]],Param.sinc[Param.MainLobe[0]:Param.MainLobe[-1]])

        sinc2=Param.sinc
        amp2=Param.amp
        freq2=Param.offset/2.0
        band2=Param.freq_re[Param.MainLobe[-1]]-Param.freq_re[Param.MainLobe[0]]

        plt.plot(sinc2,color='g',label=u'Aproximación final')
        plt.plot(Param.fty,color='k',label=u'Aproximación por picos')

        print 'Amplitude: ', amp2
        print 'Frequency: ', freq2
        print 'Bandwidth: ', band2



    plt.title(u'Estimación de parámetros')
    plt.xlabel(u'Índice de dato')
    plt.ylabel('Amplitud [u.a.]')
#    plt.legend()
    plt.show()


    #Si se realizaron dos aproximaciones, se verifica si éstas están separadas
    #por menos de 50 kHz y de ser así, se entregan los parámetros considerando
    #que se trata de una misma señal
    if entry==2:
        if abs(freq-freq2) <= 50e3:
            amp=max(amp,amp2)
            freq=abs(freq-freq2)/2.0+min(freq,freq2)
            band=band+band2
            print 'Ambas aproximaciones pertenecen a una misma señal'
            print 'Amplitude: ', amp
            print 'Frequency: ', freq
            print 'Bandwidth: ', band


    return amp, freq, band


if __name__ == '__main__':
    main()
