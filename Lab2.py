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

# --- TEST DATA SET ---
# --- Testpoints list ---
testpoints = []

pat = re.compile(r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)') #Finds patterns

# --- Formatting and cleaning test points ---
with open(testpointsfile, "r") as file:
    for line in file:
        match = pat.search(line)            # Looking for value in the parentheses (width, height)
        if not match:
            continue                    # Skips the header 
        width = float(match.group(1))           # Getting width - first value
        height = float(match.group(2))           # Getting height - second value
        testpoints.append((width, height))

testpoints = np.array(testpoints)

# --- Defining Euclidean distance for testing nearest label point  ---

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
