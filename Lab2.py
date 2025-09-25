import matplotlib.pyplot as plt
import numpy as np
import re

# --- Paths ---
datapointsfile =r"C:\Users\Pontus\python-programming-PONTUS-JOHANSSON\Labs\Lab-2\Data\datapoints.txt"
testpointsfile =r"C:\Users\Pontus\python-programming-PONTUS-JOHANSSON\Labs\Lab-2\Data\testpoints.txt"

# --- DATA POINTS ---
# --- Import datapoints as list ---
datapoints = []

# --- Open and clean data ---
with open(datapointsfile, "r") as file: 
    next(file) # Skip first line
    for line in file:
        width, height, label = line.strip().split(",")
        datapoints.append((float(width), float(height), int(label)))
        
print(datapoints)

data = np.array(datapoints) # Making a Numpy array

# --- Seperate Pichu and Pikachu for making it easier to read out ---
pichu = data[data[:, 2] == 0]   # all line with label = 0
pikachu = data[data[:, 2] == 1] # all line with label = 1

# --- Scatter plot ---
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu")
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu")
plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Pichu vs Pikachu data points")
plt.legend()
plt.show()