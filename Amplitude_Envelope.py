#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:42:18 2017

@author: Veronica Toro

Description:    This code works as a library to calculate a peak approximation
                of the input signals. It evaluates windows of c data and takes 
                two peaks within for a linear interpolation.
                Returns a vector of the same length as the input signal which 
                contains the approximation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Amplitude_Envelope():

    def __init__(self, fy, c=20):
        self.fy = fy
        self.c = c
        self.Amp_env_peak=np.zeros(1)

    def Amp_env(self):
        ae=abs(self.fy)
        maxs=np.zeros(int(len(ae)/float(self.c))+2)
        maxs[0]=ae[0]
        self.Amp_env_peak[0]=ae[0]
        i=1
        for j in np.arange(1,int(len(ae)),self.c):
            maxs[i]=ae[j:j+self.c].argmax()+j
            fx=interp1d([maxs[i-1],maxs[i]],[ae[maxs[i-1]],ae[maxs[i]]])
            xa=np.arange(maxs[i-1],maxs[i])
            ynew=fx(xa)
            self.Amp_env_peak=np.concatenate([self.Amp_env_peak,ynew])
            i+=1
        self.Amp_env_peak=np.delete(self.Amp_env_peak,0)
        j+=1
        maxs[i]=ae[-1]
        fx=interp1d([maxs[i-1],maxs[i]],[ae[maxs[i-1]],ae[maxs[i]]],bounds_error=False,fill_value="extrapolate")
        xa=np.arange(maxs[i-1],len(ae))
        ynew=fx(xa)
        self.Amp_env_peak=np.concatenate([self.Amp_env_peak,ynew])
        return self.Amp_env_peak


    def Graphic(self):
        fig, ax = plt.subplots()
        ax.plot(abs(self.fy),'red',label=u'Signal')
        ax.plot(self.Amp_env_peak,'k--',label=u'Peak approximation')

#        np.save('Amp_env.npy',Amp_env_peak[0:int(len(Amp_env_peak)/2.0)])

        plt.title('Peak Approximation')
        plt.ylabel('Amplitude [a.u.]')
        plt.xlabel('Data number')
        plt.legend()
        plt.show()
