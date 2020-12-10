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
    # Delete last element from array to keep array size the same
    SMA_12 = numpy.delete(SMA_12, -1, axis=0)
    return SMA_12

def sma26(close):
    SMA_26 = talib.SMA(close, 26)
    # Shift numpy array by adding 1 NaN to the beginning
    SMA_26 = numpy.insert(SMA_26, 0, numpy.NaN, axis=0)
    # Delete last element from array to keep array size the same
    SMA_26 = numpy.delete(SMA_26, -1, axis=0)
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

# Individual representation. (10%)
# --------------------------------
def initialise(population_size):
    population = []
    for i in range(population_size):
        weight = []
        for j in range(4):
            weight.append(random.random())
        population.append(weight)
    return population

# Fitness function. (25%)
# -----------------------
def evaluate(population, close):
    # Get trading signals. 
    sma_action = smaAction(close)
    tbr_action = tbrAction(close)
    vol_action = volAction(close)
    mom_action = momAction(close)

    # Initial budget of £3000 and stock amount of 0
    initial_budget = 3000

    # Initialise fitness
    fitness = []

    for idx, individual in enumerate(population):
        # weight of actions of each individual
        weights = population[idx]
        weighted_action = []

        for sma, tbr, vol, mom in zip(sma_action, tbr_action, vol_action, mom_action):
            action = generate_weighted_action(sma, tbr, vol, mom, weights)
            weighted_action.append(action)
        
        fitness.append(trade(close, weighted_action, initial_budget))
    return fitness

def generate_weighted_action(sma, tbr, vol, mom, weights):
    buy = 0
    sell = 0
    hold = 0

    # If any action is N/A
    if sma == 'N/A' or tbr == 'N/A' or vol == 'N/A' or mom == 'N/A':
        return 'N/A'

    if sma == 0: # hold
        hold += weights[0]
    elif sma == 1: # buy
        buy += weights[0]
    elif sma == 2: # sell
        sell += weights[0]

    if tbr == 0: # hold
        hold += weights[1]
    elif tbr == 1: # buy
        buy += weights[1]
    elif tbr == 2: # sell
        sell += weights[1]

    if vol == 0: # hold
        hold += weights[2]
    elif vol == 1: # buy
        buy += weights[2]
    elif vol == 2: # sell
        sell += weights[2]

    if mom == 0: # hold
        hold += weights[3]
    elif mom == 1: # buy
        buy += weights[3]
    elif mom == 2: # sell
        sell += weights[3]

    if buy > sell:
        if buy > hold:
            # buy action
            return 1
        else:
            # hold action
            return 0
    elif sell > hold:
        # sell action
        return 2
    else:
        #hold action
        return 0

def trade(close, action, initialBudget):
    budget = initialBudget
    portfolio = 0
    for c, a in zip(close, action):
        if a == 0 or a == 'N/A':
            continue
        elif a == 1:
            portfolio = portfolio + (budget // c)
            budget = budget % c
        elif a == 2:
            budget = budget + (portfolio * c)
            portfolio = 0
    finalClosingPrice = close[-1]
    final_budget = budget + (portfolio * finalClosingPrice)
    return final_budget

# Fitness function. (25%)
# -----------------------
def select(selection_method, tournament_size, population_size, fitness):
    parent = -1
    if selection_method == 'tournament':
        # Select participants
        participants = []
        for i in range(tournament_size):
            participant = random.randrange(population_size)
            while participant in participants:
                participant = random.randrange(population_size)
            participants.append(participant)
        # Select parent individual
        winner = participants[0]
        for i in range(tournament_size):
            if fitness[participants[i]] > fitness[winner]:
                winner = participants[i]
        parent = winner
    return parent

def run():
    # Loads closing price into NumPy array. 
    unilever = genfromtxt('Unilever.csv')

    # Loads GA configurations
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

    # Evaluates the population
    fitness = evaluate(population, unilever)
    
    parent = select(selection_method, tournament_size, population_size, fitness)
    print(parent)

if __name__ == "__main__":
    run()