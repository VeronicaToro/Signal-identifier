#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:42:18 2017

@author: Veronica Toro

Description:    Uploads a file containing data for testing propouses and
                calculates the amplitude, frecuency, bandwidth and modulation
                of the signal. There are only two posible modulations so far:
                BPSK and BFSK.
"""
import numpy as np
from Parameters import main
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# The following files are contained in the repository and are meant to be used
# for testing the algorithms.

#fy = np.load('BPSK_Dataset.npy')
fy = np.load('BFSK_Dataset.npy')

# fy contains 500 rows which are 500 signals with different SNRs. Here we take the
# 400th row. The data to be processed by the Parameters class is the FFT of the signal.

f=fy[400,1:]
ffty=abs(np.fft.fft(f))
main(ffty)     # Here is obtained and printed the amplitude, frecuency and bandwidth of the signal

mlp=joblib.load('SVM_lin_LinearKernel_C1.pkl') # Uploads an already trained Support Vector Machine model

# Data pre-processing
Hil=hilbert(f)
instphase=np.unwrap(np.angle(Hil))
std=np.std(instphase)
X=np.array([std,0])
X=X.reshape(1,-1)

R=mlp.predict(X)
if R==0:
    print 'The signal has a BFSK modulation'
else:
    print 'The signal has a  BPSK modulation'
