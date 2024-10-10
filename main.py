import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Example: Assuming 139 heroes in total
num_heroes = 139
new_match = {
    "radiant_team": [101, 50, 2, 48, 104],  # Radiant heroes
    "dire_team": [23, 86, 3, 137, 11],   # Dire heroes
}

# One-hot encode hero picks
def one_hot_heroes(radiant_team, dire_team, num_heroes):
    heroes_vector = np.zeros(num_heroes * 2)  # 139 for Radiant, 139 for Dire
    for hero in radiant_team:
        heroes_vector[hero] = 1  # Radiant heroes in the first 139 positions
    for hero in dire_team:
        heroes_vector[num_heroes + hero] = 1  # Dire heroes in the next 139 positions
    return heroes_vector

# Initialize lists to store features and labels
X = []
y = []

# Loop through each match in the JSON data
for match in data:
    radiant_team = match['radiant_team']
    dire_team = match['dire_team']
    radiant_win = match['radiant_win']
    
    # One-hot encode the hero picks
    heroes_vector = one_hot_heroes(radiant_team, dire_team, num_heroes)
    
    # Append the feature vector to X (only hero picks now)
    X.append(heroes_vector)
    
    # Append the label (radiant_win) to y (1 for win, 0 for loss)
    y.append(1 if radiant_win else 0)

# Convert lists to numpy arrays for use in machine learning models
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Predict the outcome of the new match based on hero picks
z_heroes_vector = one_hot_heroes(new_match['radiant_team'], new_match['dire_team'], num_heroes)
prediction = model.predict([z_heroes_vector])

# Interpret the prediction
if prediction == 1:
    print("Radiant is predicted to win!")
else:
    print("Radiant is predicted to lose!")

# Evaluate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
