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

# --- Creating input with error handeling ----

def getUserInput(prompt): 
    while True:
        answer = input(prompt).strip() # Takes away the space 
        answer = answer.replace(",", ".") # Letting the user write 25,0 and conver it to 25.0
        try:
            value = float(answer)
        except ValueError:
            print("Error: Enter a numeric value(ex. 22.5). Please try again.")
            continue

        # Not NaN/inf
        if not np.isfinite(value):
            print("Error: Value must be finite (not NaN/inf). Please try again.")
            continue

        if value <= 0:
            print("Error: Value must be > 0, cannot be a negative number. Please try again.")
            continue

        return value

def UserInput():
    width  = getUserInput("Enter width (cm): ")
    height = getUserInput("Enter height (cm): ")
    return np.array([width, height])

# Read one user point
user_point = UserInput()
print("Your points:", user_point)
print(f"{user_point[0]:.1f}, {user_point[1]:.1f}")  

# --- Test the point --- 
classified_Userpoint = []
for i, test in enumerate(user_point):
    
# --- Classify the user point with 1-NN ---
    distances = [(euclidean_distance(user_point, tr[:2]), int(tr[2])) for tr in data]
    nearest_label = min(distances, key=lambda x: x[0])[1]
    predicted = "Pichu" if nearest_label == 0 else "Pikachu"
    print(f"User point -> {user_point[0]:.1f}, {user_point[1]:.1f} classified as: {predicted}")

# Keep a list (useful if you later add more user points)
classified_Userpoint = [(user_point[0], user_point[1], predicted)]

# --- Plot ---
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu (train)")
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu (train)")

# Add the user point
shown = set()
for (width, height, lab) in classified_Userpoint:
    color = "#FFF06A" if lab == "Pichu" else "#E19720"
    lbl = f"User points ({lab})" if lab not in shown else "_nolegend_"
    plt.scatter(width, height, color=color, marker="o", s=100, edgecolors="red", label=lbl)
    shown.add(lab)

plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Classification of User points (1-NN)")
plt.legend()
plt.show()

# --- Define KKN-10 ---
def classify_knn(classified_Userpoint, datapoints, k=10):
    distances = []
    for train in datapoints:
        dist = euclidean_distance(classified_Userpoint, train[:2])
        distances.append((dist, int(train[2])))

    # sort by distance and take the k closest
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    # count how many are Pichu (0) and Pikachu (1)
    labels = [label for _, label in k_nearest]
    pichu_count = labels.count(0)
    pikachu_count = labels.count(1)

    return "Pichu" if pichu_count > pikachu_count else "Pikachu"

# Testing with our four test points
for i, test in enumerate(classified_Userpoint):
    predicted = classify_knn(test, data, k=10)
    print(f"Test point {i+1} {tuple(test)} classified as: {predicted} (k=10)")

classified_points_k10 = []
for test in classified_Userpoint:
    predicted = classify_knn(test, data, k=10)
    classified_points_k10.append((test[0], test[1], predicted))

# Scatter plot with KKN-10 with test points
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu (train)")
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu (train)")

# --- Putting in user point nearest neaber data in plot with stars ---
shown = set()  # Keeps track of which classes are already in the scatter plot

# --- For Loop to add user point with closest data point label --- 
for (width, height, lab) in classified_points_k10:
    color = "#FFF06A" if lab == "Pichu" else "#E19720"
    # Giving correct color depending on value
    # Using "_nolegend_" so we don't get duplicates. And shortened label to lbl
    lbl = f"User points ({lab})" if lab not in shown else "_nolegend_"

    plt.scatter(width, height,
                color=color, marker="o", s=100, edgecolors="black",
                label=lbl)

    # Add test data point to scatter plot
    shown.add(lab)

plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Classification of user points (k=10)")
plt.legend()
plt.show()

# --- 100 TRAIN & 50 TEST DATA POINT 
# --- Randomnize pichu and pikachu so we don't use the same
np.random.seed(42)  # Splitting
pichu_shuffled   = pichu[np.random.permutation(len(pichu))]
pikachu_shuffled = pikachu[np.random.permutation(len(pikachu))]

# Making copies of train and test data
train_data = np.vstack([pichu_shuffled[:50], pikachu_shuffled[:50]])
test_data  = np.vstack([pichu_shuffled[50:75], pikachu_shuffled[50:75]])

# Shuffle the train and test data set so pichu and pikachu does not come in order
train_data = train_data[np.random.permutation(len(train_data))]
test_data  = test_data[np.random.permutation(len(test_data))]

print("Train shape:", train_data.shape)  # (100, 3)
print("Test shape:", test_data.shape)    # (50, 3)

#Visulize the data

plt.scatter(train_data[train_data[:, 2] == 0][:, 0], train_data[train_data[:, 2] == 0][:, 1], 
            color="#E6CE2B", label="Pichu (train)", alpha=0.6)
plt.scatter(train_data[train_data[:, 2] == 1][:, 0], train_data[train_data[:, 2] == 1][:, 1], 
            color="#E19720", label="Pikachu (train)", alpha=0.6)

plt.scatter(test_data[test_data[:, 2] == 0][:, 0], test_data[test_data[:, 2] == 0][:, 1], 
            color="#2F3336", marker="x", label="Pichu (test)")
plt.scatter(test_data[test_data[:, 2] == 1][:, 0], test_data[test_data[:, 2] == 1][:, 1], 
            color="#F62D14", marker="x", label="Pikachu (test)")

plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Train vs Test data (Pichu & Pikachu)")
plt.legend()
plt.show()