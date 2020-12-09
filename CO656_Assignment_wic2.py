# Computational Intelligence in Business, Economics & Finance Assignment
# Optimising trading strategies with technical analysis
# 
# Author: Wai Ip Chu  
# Login: wic2  
# Created: 23.11.2020
# 
# For information on how to run script, follow instructions in README.md

import sys
import numpy
from numpy import genfromtxt
import talib
import math

numpy.set_printoptions(threshold=sys.maxsize)

# Part A: Implementing technical indicators and trading signals. 
# ==============================================================


# Task 1: Technical Indicators. (10%)
# -----------------------------------

# Task 1a: Calculate the 12 days moving average and 26 days moving average. 
def sma12(close):
    sma12 = talib.SMA(close, 12)
    # Shift numpy array by adding 1 NaN to the beginning
    sma12 = numpy.insert(sma12, 0, numpy.NaN)
    return sma12

def sma26(close):
    sma26 = talib.SMA(close, 26)
    # Shift numpy array by adding 1 NaN to the beginning
    sma26 = numpy.insert(sma26, 0, numpy.NaN)
    return sma26

# Task 1b: Calculate the 24 days trade break rule. 
def tbr24(close):
    # TODO
    pass

# Task 1c: Calculate the 29 days volatility. 
def vol29(close):
    # TODO
    pass

# Task 1d: Calculate the 10 days momentum. 
def mom10(close):
    mom10 = talib.MOM(close, 10)
    return mom10


# Task 2: Trading signals. (10%)
# ------------------------------

# Task 2a: Use the two SMA to generate signals. 
# If SMA_12 > SMA_26 => 1 (buy)
# If SMA_12 < SMA_26 => 2 (sell)
# If SMA_12 = SMA_26 => 0 (hold)
def smaAction(close):
    SMA_12 = sma12(close)
    SMA_26 = sma26(close)
    sma_action = []

    for x, y in zip(SMA_12, SMA_26):
        if math.isnan(x) or math.isnan(y):
            sma_action.append('N/A')
        elif x == y:
            sma_action.append(0)
        elif x > y:
            sma_action.append(1)
        else:
            sma_action.append(2)
    return sma_action
    
# Task 2b: Use TBR to generate signals. 
# If TBR_24 > –0.02 => 2 (sell)
# If TBR_24 < –0.02 => 1 (buy)
# If TBR_24 = –0.02 => 0 (hold)
def tbrAction(close):
    TBR_24 = tbr24(close)
    print(TBR_24)
    tbr_action = []

    for x in TBR_24:
        if math.isnan(x):
            tbr_action.append('N/A')
        elif x == -0.02:
            tbr_action.append(0)
        elif x > -0.02:
            tbr_action.append(1)
        else:
            tbr_action.append(2)
    return tbr_action

# Task 2c: Use VOL to generate signals. 
# If VOL_29 > 0.02 => 1 (buy)
# If VOL_29 < 0.02 => 2 (sell)
# If VOL_29 = 0.02 => 0 (hold)
def volAction(close):
    VOL_29 = vol29(close)
    print(VOL_29)
    vol_action = []

    for x in VOL_29:
        if math.isnan(x):
            vol_action.append('N/A')
        elif x == 0.02:
            vol_action.append(0)
        elif x > 0.02:
            vol_action.append(1)
        else:
            vol_action.append(2)
    return vol_action

# Task 2d: Use MOM to generate signals. 
# If MOM_10 > 0 => 1 (buy)
# If MOM_10 < 0 => 2 (sell)
# If MOM_10 = 0 => 0 (hold)
def momAction(close):
    MOM_10 = mom10(close)
    print(MOM_10)
    mom_action = []

    for x in MOM_10:
        if math.isnan(x):
            mom_action.append('N/A')
        elif x == 0:
            mom_action.append(0)
        elif x > 0:
            mom_action.append(1)
        else:
            mom_action.append(2)
    return mom_action

def run():
    # Loads data into NumPy array. 
    unilever = genfromtxt('Unilever.csv')
    
    # Get trading signals. 
    sma_action = smaAction(unilever)
    print(sma_action)
    # tbr_action = tbrAction(unilever)
    # vol_action = volAction(unilever)
    # mom_action = momAction(unilever)

if __name__ == "__main__":
    run()