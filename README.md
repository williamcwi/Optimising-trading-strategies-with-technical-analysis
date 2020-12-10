Computational Intelligence in Business, Economics & Finance Assignment
======================================================================

Author: Wai Ip Chu  
Login: wic2  
Created: 23.11.2020  

<br />

---
## Installation
---

Create a virtual environment and install numpy and TA_Lib using command

```
$ pip install -r requirements.txt
```

If you have trouble installing TA-Lib, you can download the .whl [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and use command

```
$ pip install TA_Lib-file-name.whl
```

<br />

---
## Configurations
---
The file GA_config.json include configurations for the Genetic Algorithm:  

<br />

```
"population_size": 100
```
Total number of individuals in each generation

<br />

```
"max_generation": 50
```
Termination criteria for maximum number of generations

<br />

```
"selection_method": "tournament"
```
Selection method for individuals to produce offspring (the GA should use tournament selection)

<br />

```
"tournament_size": 5
```
Number of individuals selected for the tournament

<br />

```
"crossover_method": "one-point"
```
Crossover operator method:
- one-point
- two-point

<br />

```
"mutation_method": "point"
```
Mutation operator method: 
- point
- bit-string

<br />

```
"chance_of_mutation": 0.05
```
Chance of mutation happening