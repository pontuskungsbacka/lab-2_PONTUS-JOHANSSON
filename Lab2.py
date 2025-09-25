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

# --- Classify - test points with 1-nn ---
classified_points = []
for i, test in enumerate(testpoints):
    distancesTestpoints = [] # Creating a list for distance
    for train in data:
        dist = euclidean_distance(test, train[:2])
        distancesTestpoints.append((dist, int(train[2])))
    #print(distances) check whats in the list
    # Sort distance with closest label 0 or 1
    distancesTestpoints.sort(key=lambda x: x[0])
    nearest_label = distancesTestpoints[0][1]
    predicted = "Pichu" if nearest_label == 0 else "Pikachu"
    classified_points.append((test[0], test[1], predicted))


    # Write out distances with closest label 
    print(f"Test points {i+1} with (width, height): {test[0]}, {test[1]} predicted as: {predicted}")
    
# --- Make a new graf for checking the test data ---
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu (train)")
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu (train)")

# --- Putting in test point nearest neaber data in plot with stars ---
shown = set()  # Keeps track of which classes are already in the scatter plot

# --- For Loop to add test point with closest data point label --- 
for (width, height, lab) in classified_points:
    color = "#FFF06A" if lab == "Pichu" else "#E19720"
    # Giving correct color depending on value
    # Using "_nolegend_" so we don't get duplicates. And shortened label to lbl
    lbl = f"Test point ({lab})" if lab not in shown else "_nolegend_"

    plt.scatter(width, height,
                color=color, marker="*", s=250, edgecolors="black",
                label=lbl)

    # Add test data point to scatter plot
    shown.add(lab)

plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Classification of test points (1-NN)")
plt.legend()
plt.show()
