# Signal-identifier
Algorithm in Python 2.7 for amplitude, frequency, bandwidth and modulation identification of a signal

In this repository there are two data files:

- BFSK_Dataset.npy
- BPSK_Dataset.npy

Each of these files contains 500 simulated signals composed by 1024 time-sampled data. When these files are opened it is returned a 2D matrix of size (500 x 1025) as the first column content the SNR with which the signal, in the corresponding row, was calculated. The signals were created following and modifying the algorithms created by Tim O'Shea that can be found in https://github.com/radioML/dataset. 20 SNR values were used. These range from -20 dB to 18 dB in steps of 2 dB so there are 25 rows for each SNR value. That is, rows from 0 to 24 correspond to signals calculated using a -20 dB SNR, rows from 25 to 49 are signals with a -18 dB value, and so on.

These datasets were used in the training phase of the Artificial Neural Networks used for the modulation classification.

# Prerequisites
To run this you must have scikit-learn version 0.18.1 and joblib version 0.11. You can install them by running on your Ubuntu console
```
sudo pip install scikit-learn==0.18.1
```
```
sudo pip install joblib==0.11
```
