#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 1 11:56:24 2017

@author: Veronica Toro

Description:    This code makes use of different models to approximate the amplitude,
                frequency and bandwith of a signal by analysing its FFT.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from Amplitude_Envelope import Amplitude_Envelope
import random

class Parameters():

    def __init__(self,entry,_ffty):
        self.entry=entry

        # Only the positive frequencies are taken
        _fyc=np.zeros(int(len(_ffty)/2.0))
        if entry==1:    # This is the first approximation
            _fc=abs(_ffty[:int(len(_ffty)/2)])
            
            # In order to somilate a frequency offset in the BPSK signals,
            # a shift FFT was done and it was centered in a random value
            PSK=0   # Set to 1 when using BPSK signals from the provided datasets
            if PSK:
                idx=int(len(_fc)/2.0)
                random.seed(21)
                idx=random.randrange(100, 400, 1)   # This range is given becuase the signal has 512 data points
                _fyc[:idx]=_fc[-idx:]
                _fyc[idx:]=_fc[:-idx]
            else:
                _fyc=_fc
            plt.plot(_fyc, color='r' ,label=u'Señal')
            
        elif entry==2:    # This is the second approximation
            _fyc=_ffty

        AE=Amplitude_Envelope(_fyc,20)  # Initializes the peak approximation with 20-points-windows
        self.fty=AE.Amp_env()           # In self.fty is kept the peak approximation
#        AE.Graphic()                   # Plots the peak approximation
        _Fs=200000.0                    # Sample rate at which the provided datasets were calculated
        self.L=float(len(self.fty))
        self.Lint=len(self.fty)
        self.freq=np.arange(0,_Fs,_Fs/(self.L),dtype='float')              # Vector containing the frequencies of the complete FFT
        self.freq_re=np.arange(0,_Fs/2.0,_Fs/(2.0*self.L),dtype='float')   # Vector containing the positive frequencies of the FFT

        self.off=self.fty.argmax()      # The initial offset is located at the maximum point of the FFT
        self.offset=self.freq[self.off]
        self.amp=np.max(self.fty)       # The initial amplitude is the maximum value of the FFT
        self.width=0.1      # This parameter was set because is small enough to be increased latter in the width subroutine
                            # and evaluate different errors
        self.alpha=self.width/self.L
        self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset))) # Initial sinc approximation
        _gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)   # Finds the right end of the main lobe of the sinc approx
        _gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)  # Finds the left end of the main lobe of the sinc approx
        self.MainLobe=np.arange(_gm,self.off+_gp,1)                         # Points in the main lobe


    def Amplitud(self):
        # It is defined the main lobe again
        gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
        gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
        self.MainLobe=np.arange(gm,self.off+gp,1)
        
        landa=0.5   # Smoother parameter
        # The stop of the routine was defined by a mean squared error
        Err=100
        while Err>=10:
            self.sinc=abs(np.sinc(self.alpha*(self.freq-self.offset)))
            V=abs(self.fty-self.sinc)   # Residual function which is the difference between the signal and the approximation
            num=0
            den=0
            for i in self.MainLobe:
                num=num+V[i]*self.sinc[i]/self.amp
                den=den+(self.sinc[i]/self.amp)**2
            deltaAmp=num/den
            self.amp=(1-landa)*self.amp+landa*deltaAmp
            
            # Normalized mean squared error
            Err=(mse(self.fty[self.MainLobe[0]:self.MainLobe[-1]],self.sinc[self.MainLobe[0]:self.MainLobe[-1]]))/self.amp
        self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset)))     # Updates the sinc approximation


    def Offset(self):
        j=0
        deltaOffset=np.zeros(2)
        # This routine runs until the offset moves one unit to a direction and then returns
        while deltaOffset[-1]*deltaOffset[-2]>=0:
            # It is defined the main lobe again
            gp=find_nearest(self.alpha*(self.freq[self.off:]-self.offset),1)
            gm=find_nearest(self.alpha*(self.freq[:self.off]-self.offset),-1)
            self.MainLobe=np.arange(gm,self.off+gp,1)
            
            Lmain=len(self.MainLobe)
            V=abs(self.fty-self.sinc)
            cl=np.min([np.max(V[self.MainLobe[0]:self.off]),np.max(self.sinc[self.MainLobe[0]:self.off])])
            
            # The "mass" produced by the left side of the main lobe, taking the offset of the sinc as the pivot
            ML=0
            for i in range(self.MainLobe[0],self.off+1):
                ML=ML+np.clip(V[i],0,cl)-np.clip(self.sinc[i],0,cl)
            ML=ML/Lmain
            cl=np.min([np.max(V[self.off:self.MainLobe[-1]]),np.max(self.sinc[self.off:self.MainLobe[-1]])])
            
            # The "mass" produced by the right side of the main lobe, taking the offset of the sinc as the pivot
            MR=0
            for i in range(self.off,self.MainLobe[-1]+1):
                MR=MR+np.clip(V[i],0,cl)-np.clip(self.sinc[i],0,cl)
            MR=MR/Lmain
            
            # If the left side procudes a bigger torque, a step to the left is taken
            if ML-MR > 0:
                deltaOffset[j]=-1
            else:
                deltaOffset[j]=1
            self.offset=self.offset+deltaOffset[j]
            self.sinc=self.amp*abs(np.sinc(self.alpha*(self.freq-self.offset)))
            self.off=self.sinc.argmax()
            j+=1
            deltaOffset=np.resize(deltaOffset,j+1)

    def Width(self):
        beta=1
        j=1
        Errors=np.array([2e3,1e3,1])
        Alphas=np.zeros(3,dtype='float')
        # The stop of the routine was defined by a mean squared error
        for i in range(50):     # Runs 50 times
            # It is defined the main lobe again
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
            
            # Normalized mean squared error
            Err=(mse(self.fty[self.MainLobe[0]:self.MainLobe[-1]],self.sinc[self.MainLobe[0]:self.MainLobe[-1]]))/self.amp
            j+=1
            Errors=np.resize(Errors,j+1)
            Errors[j]=Err
            Alphas=np.resize(Alphas,j+1)
            Alphas[j]=self.alpha
            
        # It is taken the width at which the error was minimum
        idx=np.argmin(Errors)
        self.sinc=self.amp*abs(np.sinc(Alphas[idx]*(self.freq-self.offset)))
        
        # It is not necessary to keep these vectors
        Errores=[]
        Alphas=[]

# This function computes the nearest point to 'value' in 'array'
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def main(ffty,Parametros_cls=Parametros, options=None):
#def main(Parametros_cls=Parametros, options=None):    # In case you want to run it alone and not from an external code

    # These are the provided dataset
#    fy = np.load('BPSK_Dataset.npy')
#    fy = np.load('BFSK_Dataset.npy')

    # Only one row is taken at a time. In this case we use the 400th row and the input data
    # for the Parameters class is the FFT of the signal.
#    ffty=abs(np.fft.fft(fy[400,1:]))

    #entry=1 is for the first approximation
    entry=1
    Param = Parametros_cls(entry,ffty)  # Initializes the class
    
    # An initial error is defined and it begins an interation over different parameters
    Err=100000.0
    while Err>0.05:
        Param.Amplitud()
        Param.Offset()
        Param.Width()
        Err=(mse(Param.fty[Param.MainLobe[0]:Param.MainLobe[-1]],Param.sinc[Param.MainLobe[0]:Param.MainLobe[-1]]))/Param.amp
    
    # The remaining mean squared error is calculated to determine if another approximation is necessary
    ErrTodo=(mse(Param.fty,Param.sinc))/Param.amp

###############################################################################
#                               Amplitud
#   (Here it is used a reception-power characterization of an Ettus N210)
#                   You woudn't probably need this
###############################################################################
    Carac=np.load('Rx_Power_Characterization_Gain_0_Regression.npy')
    if ((Param.offset/2.0)>=144e6) and ((Param.offset/2.0)<=148e6):
        # VHF
        idx=50e3    # Frequency step
        x1=Carac[int(2e6/idx):int(2e6/idx)+20,0]
        y1=Carac[int(2e6/idx):int(2e6/idx)+20,1]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    elif ((Param.offset/2.0)>=430e6) and ((Param.offset/2.0)<=440e6):
        # UHF
        fact=1600
        idx=50e3    # Frequency step
        x1=Carac[fact+int(5e6/idx):fact+int(5e6/idx)+20]
        y1=Carac[fact+int(5e6/idx):fact+int(5e6/idx)+20]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    elif ((Param.offset/2.0)>=2.4e9) and ((Param.offset/2.0)<=2.5e9):
        # S-band
        fact=5600
        idx=1e6     # Frequency step
        x1=Carac[fact+int(500e6/idx):fact+int(500e6/idx)+20]
        y1=Carac[fact+int(500e6/idx):fact+int(500e6/idx)+20]
        fx=interp1d(x1,y1,bounds_error=False,fill_value="extrapolate")
        Amp=fx(Param.amp)
    else:
        Amp=Param.amp
###############################################################################


    # Results are taken and printed
    sinc=Param.sinc
    amp=Param.amp
    freq=Param.offset/2.0
    band=Param.freq_re[Param.MainLobe[-1]]-Param.freq_re[Param.MainLobe[0]]

    print 'Amplitude: ', Amp
    print 'Frequency: ', freq
    print 'Bandwidth: ', band

    plt.plot(sinc,color='g',label=u'Aproximación final')
    plt.plot(Param.fty,color='k',label=u'Aproximación por picos')

    # If the remaining error is bigger than 0.007, then another approximation takes place
    if ErrTodo >= 0.007:
        entry=2     # Second approximation is set
        # The input signal for this second approx. is the residual function
        fy=abs(Param.fty-sinc)
        Param = Parametros_cls(entry,fy)
        Err=100000.0
        while Err>0.05:
            Param.Amplitud()
            Param.Offset()
            Param.Width()
            Err=mse(Param.fty[Param.MainLobe[0]:Param.MainLobe[-1]],Param.sinc[Param.MainLobe[0]:Param.MainLobe[-1]])

        # Results are taken and printed
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


    # In case that two approximations were made, it is calculated whether the center frequencies
    # are separated as much as 50 kHz. If so, it is considered that they belong to the same signal
    # so the given parameters are an average of the already calculated parameters 
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
