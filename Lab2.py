import matplotlib.pyplot as plt
import numpy as np
import re

"""
Lab 2 –  (Pichu vs Pikachu)
- Load training data (width, height, label) from file
- Parse test points from a formatted text file "(x, y)"
- Classify with 1-NN and k-NN (k=10), visualize results
- Build a stratified train/test split (100/50) and evaluate accuracy
- Repeating previously step 10 times, calculating accuracy over 10 times

Project made by Pontus Johansson AIM25G
A light guidance has been done with ChatGPT to understand the structure and troubleshooting,
but I have programmade as I want to build the application up with Pichu vs Pikachu. Also help from W3school, 
medium.com(K-Nearest Neighbors) and searches on the internet with suggestions 
from Copilot have helped me to complete the task.
"""

# --- Paths ---
datapointsfile =r"C:\Users\Pontus\python-programming-PONTUS-JOHANSSON\Labs\Lab-2\Data\datapoints.txt"
testpointsfile =r"C:\Users\Pontus\python-programming-PONTUS-JOHANSSON\Labs\Lab-2\Data\testpoints.txt"

# --- DATA POINTS ---
# Load datapoints into a list of tuples: (width: float, height: float, label: int)
datapoints = []

# Read file, skip header, parse numeric types (float, float, int)
with open(datapointsfile, "r") as file: 
    next(file) # Skip first line
    for line in file:
        width, height, label = line.strip().split(",")
        datapoints.append((float(width), float(height), int(label)))
        
# print(datapoints) print out the list

data = np.array(datapoints) # # Convert to NumPy array for vectorized operations

# Split by class label for plotting convenience
pichu = data[data[:, 2] == 0]   # all line with label = 0
pikachu = data[data[:, 2] == 1] # all line with label = 1

# --- Scatter plot of training data from data points ---
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu")
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu")
plt.xlabel("Width (cm)")
plt.ylabel("Height (cm)")
plt.title("Pichu vs Pikachu data points")
plt.legend()
plt.show()

# --- TEST DATA SET ---
# Container for parsed test points (width, height)
testpoints = []

pat = re.compile(r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)') # Regex: capture two floats inside parentheses, e.g., "(25, 32)" with optional spaces/signs // Copilot suggestion 
# --- Reading and cleaning test points ---
with open(testpointsfile, "r") as file:
    for line in file:
        match = pat.search(line)            # Looking for value in the parentheses (width, height)
        if not match:
            continue                    # Skips the header 
        width = float(match.group(1))           # Getting width - first value
        height = float(match.group(2))           # Getting height - second value
        testpoints.append((width, height))

testpoints = np.array(testpoints)

# Euclidean distance in 2D (width, height)

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Classify test points with 1-NN
classified_points = []
for i, test in enumerate(testpoints):
    distancesTestpoints = [] # Creating a list for distance
    for train in data:
        dist = euclidean_distance(test, train[:2])
        distancesTestpoints.append((dist, int(train[2])))
    # Sort by distance (ascending) and take nearest label (0 or 1)
    distancesTestpoints.sort(key=lambda x: x[0])
    nearest_label = distancesTestpoints[0][1]
    predicted = "Pichu" if nearest_label == 0 else "Pikachu"
    classified_points.append((test[0], test[1], predicted))


    # Log classification result for this test point
    print(f"Test points {i+1} with (width, height): {test[0]}, {test[1]} predicted as: {predicted}")
    
# --- Plot training data and classified test points ---
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu (train)")
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu (train)")

# Add classified test points (stars); use a set to avoid duplicate legend entries
shown = set()

"""Inspired by ChatGPT: use a set + "_nolegend_" to avoid duplicate legend entries per class."""

# Plot each classified test point; label only the first occurrence per class
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

# --- Collect user input with validation (numeric, finite, > 0) ---
# Based on ChatGPT advice: robust numeric input (supports comma decimal, rejects NaN/inf, requires >0).
def getUserInput(prompt): 
    while True:
        answer = input(prompt).strip() # remove leading/trailing whitespace 
        answer = answer.replace(",", ".") # allow comma decimals (e.g., "25,0" -> "25.0")
        try:
            value = float(answer)
        except ValueError:
            print("Error: Enter a numeric value(ex. 22.5). Please try again.")
            continue

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

# --- Classify the single user point with 1-NN ---
classified_Userpoint = []
for i, test in enumerate(user_point):
    
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

# --- Define k-NN (k=10) classifier for a single point ---
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

# Classify the (user) point(s) with k=10 and print results
for i, test in enumerate(classified_Userpoint):
    predicted = classify_knn(test, data, k=10)
    print(f"Test point {i+1} {tuple(test)} classified as: {predicted} (k=10)")

classified_points_k10 = []
for test in classified_Userpoint:
    predicted = classify_knn(test, data, k=10)
    classified_points_k10.append((test[0], test[1], predicted))

# Scatter plot with k-NN (k=10) classification for user point(s)
plt.scatter(pichu[:, 0], pichu[:, 1], color="#FFF06A", label="Pichu (train)")
plt.scatter(pikachu[:, 0], pikachu[:, 1], color="#E19720", label="Pikachu (train)")

# Add user point(s); avoid duplicate legend entries via a set
shown = set()

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

# --- Build 100-train (50 Pichu, 50 Pikachu) and 50-test (25/25) via stratified sampling ---
# Shuffle within each class to avoid reusing the same examples
# Stratified selection by class (50/50 train, 25/25 test), idea discussed with ChatGPT.
np.random.seed(42)  # Splitting
pichu_shuffled   = pichu[np.random.permutation(len(pichu))]
pikachu_shuffled = pikachu[np.random.permutation(len(pikachu))]

# Making copies of train and test data
train_data = np.vstack([pichu_shuffled[:50], pikachu_shuffled[:50]])
test_data  = np.vstack([pichu_shuffled[50:75], pikachu_shuffled[50:75]])

# Shuffle train/test order so classes are not grouped
train_data = train_data[np.random.permutation(len(train_data))]
test_data  = test_data[np.random.permutation(len(test_data))]

print("Train shape:", train_data.shape)  # (100, 3)
print("Test shape:", test_data.shape)    # (50, 3)

# Visualize train vs test (class-colored)

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

# --- Classify with k-NN (k=10) and compute accuracy ---
# Adapted from ChatGPT: k-NN with majority vote and tie-break by nearest neighbor; returns 0/1 labels.
def classify_knn(test_point, train_data, k=10):
    dists = []
    for train in train_data:
        d = euclidean_distance(test_point, train[:2])
        dists.append((d, int(train[2])))
    dists.sort(key=lambda x: x[0])
    k_nearest = dists[:k]
    labels = [lab for _, lab in k_nearest]
    pichu = labels.count(0)
    pikachu = labels.count(1)
    if pichu > pikachu: return 0
    if pikachu > pichu: return 1
    return k_nearest[0][1]  # tie-break

def calculate_accuracy(train_data, test_data, k=10):
    correct = 0
    for row in test_data:
        pred = classify_knn(row[:2], train_data, k)
        if pred == int(row[2]):
            correct += 1
    return correct / len(test_data)

# Actual labels in the test set (0 = Pichu, 1 = Pikachu)
y_true = test_data[:, 2].astype(int)

# Predictions for all test examples at given k
def predict_all(test_data, train_data, k=10):
    return np.array([classify_knn(row[:2], train_data, k) for row in test_data])

k = 10 
y_pred = predict_all(test_data, train_data, k=k)

# Count TP, TN, FP, FN (Pikachu = positive = 1)

TP = int(np.sum((y_true == 1) & (y_pred == 1)))  # Pikachu correct
TN = int(np.sum((y_true == 0) & (y_pred == 0)))  # Pichu correct
FP = int(np.sum((y_true == 0) & (y_pred == 1)))  # Pichu -> Pikachu (FALSE)
FN = int(np.sum((y_true == 1) & (y_pred == 0)))  # Pikachu -> Pichu (FALSE)

# Accuracy = (TP + TN) / total
accuracy = (TP + TN) / y_true.size

print(f"k={k}  ->  TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy: {accuracy:.3f}")

# --- Repeat stratified split + evaluation over 10 seeds ---
# Inspired by ChatGPT: stratified split using a local RNG (reproducible seeds per run).
def make_stratified_split(data, rng, n_train_per_class=50, n_test_per_class=25):
    """Returns (train_data, test_data) with exact balance 50/50 and 25/25 respectively."""
    pichu   = data[data[:, 2] == 0]
    pikachu = data[data[:, 2] == 1]

    # Checking 
    need_per_class = n_train_per_class + n_test_per_class
    if len(pichu) < need_per_class or len(pikachu) < need_per_class:
        raise ValueError(f"Need at least {need_per_class} points per class, "
                        f"has Pichu={len(pichu)}, Pikachu={len(pikachu)}.")

    # Randomize indices within each class via rng
    p_idx = rng.permutation(len(pichu))
    k_idx = rng.permutation(len(pikachu))

    pichu_shuf   = pichu[p_idx]
    pikachu_shuf = pikachu[k_idx]

    # Pick 50/50 for training and 25/25 for testing
    p_train = pichu_shuf[:n_train_per_class]
    p_test  = pichu_shuf[n_train_per_class:n_train_per_class + n_test_per_class]

    k_train = pikachu_shuf[:n_train_per_class]
    k_test  = pikachu_shuf[n_train_per_class:n_train_per_class + n_test_per_class]

    train = np.vstack([p_train, k_train])
    test  = np.vstack([p_test,  k_test])

    # Shuffle the order
    train = train[rng.permutation(len(train))]
    test  = test[rng.permutation(len(test))]
    return train, test

# --- Parametrar ---
# Based on ChatGPT walkthrough: repeat split+evaluation over fixed seeds; plot per-run accuracy and mean line.
K = 10
SEEDS = [11,22,33,44,55,66,77,88,99,111]  # 10 different seeds

# --- Run the experiment 10 times ---
accuracies = []
for run, seed in enumerate(SEEDS, start=1):
    rng = np.random.default_rng(seed)
    train_data, test_data = make_stratified_split(data, rng, n_train_per_class=50, n_test_per_class=25)
    acc = calculate_accuracy(train_data, test_data, k=K)
    accuracies.append(acc)
    print(f"Running {run:2d} (seed={seed}): accuracy = {acc:.2f}")

# --- Aggregate results ---
accuracies = np.array(accuracies, dtype=float)
mean_acc = float(np.mean(accuracies))
std_acc  = float(np.std(accuracies, ddof=1))
min_acc  = float(np.min(accuracies))
max_acc  = float(np.max(accuracies))

print("\nSummary over 10 runs:")
print(f"Middle accuracy: {mean_acc:.3f}")
print(f"Std (spread): {std_acc:.3f}")
print(f"Min/Max: {min_acc:.3f} / {max_acc:.3f}")

# --- Plot ---
plt.figure(figsize=(7,4))
plt.plot(range(1, len(accuracies)+1), accuracies, marker="o")
plt.axhline(mean_acc, linestyle="--")
plt.xticks(range(1, len(accuracies)+1))
plt.ylim(0.0, 1.0)
plt.xlabel("Runs")
plt.ylabel("Accuracy")
plt.title(f"Accuracy over 10 runs (k={K})")
plt.show()