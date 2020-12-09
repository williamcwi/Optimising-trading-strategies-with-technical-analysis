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
import json
import random

numpy.set_printoptions(threshold=sys.maxsize)



# Part A: Implementing technical indicators and trading signals. 
# ==============================================================


# Task 1: Technical Indicators. (10%)
# -----------------------------------

# Task 1a: Calculate the 12 days moving average and 26 days moving average. 
def sma12(close):
    SMA_12 = talib.SMA(close, 12)
    # Shift numpy array by adding 1 NaN to the beginning
    SMA_12 = numpy.insert(SMA_12, 0, numpy.NaN, axis=0)
    return SMA_12

def sma26(close):
    SMA_26 = talib.SMA(close, 26)
    # Shift numpy array by adding 1 NaN to the beginning
    SMA_26 = numpy.insert(SMA_26, 0, numpy.NaN, axis=0)
    return SMA_26

# Task 1b: Calculate the 24 days trade break rule. 
def tbr24(close):
    TBR_24 = numpy.array([])
    for idx, val in enumerate(close):
        if idx < 24:
            TBR_24 = numpy.append(TBR_24, numpy.NaN)
        else:
            sliced_array = close[idx-24:idx]
            result = (val - numpy.max(sliced_array))/numpy.max(sliced_array)
            TBR_24 = numpy.append(TBR_24, result)
    return TBR_24

# Task 1c: Calculate the 29 days volatility. 
def vol29(close):
    VOL_29 = numpy.array([])
    for idx, val in enumerate(close):
        if idx < 29:
            VOL_29 = numpy.append(VOL_29, numpy.NaN)
        else:
            sliced_array_1 = close[idx-28:idx+1]
            sliced_array_2 = close[idx-29:idx]
            # population standard deviation
            result = numpy.std(sliced_array_1)/(numpy.sum(sliced_array_2)/29)
            VOL_29 = numpy.append(VOL_29, result)
    return VOL_29

# Task 1d: Calculate the 10 days momentum. 
def mom10(close):
    MOM_10 = talib.MOM(close, 10)
    return MOM_10


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



# Part B: Genetic Algorithm
# =========================


# Implement GA to evolve a set of weights to determine optimal trading action. (60%)
# ----------------------------------------------------------------------------------

def initialise(population_size):
    population = []
    for i in range(population_size):
        weight = []
        for j in range(4):
            weight.append(random.random())
        population.append(weight)
    return population

def evaluate():
    pass



def run():
    # Loads data into NumPy array. 
    unilever = genfromtxt('Unilever.csv')

    # Loads JSON.
    with open('GA_config.json') as json_file:
        config = json.load(json_file)
        population_size = config['population_size']
        max_generation = config['max_generation']
        selection_method = config['selection_method']
        crossover_method = config['crossover_method']
        mutation_method = config['mutation_method']
        chance_of_mutation = config['chance_of_mutation']
        tournament_size = config['tournament_size']
    
    # Initialises the population
    population = initialise(population_size)

    evaluate()

    # Get trading signals. 
    # sma_action = smaAction(unilever)
    # tbr_action = tbrAction(unilever)
    # vol_action = volAction(unilever)
    # mom_action = momAction(unilever)

if __name__ == "__main__":
    run()