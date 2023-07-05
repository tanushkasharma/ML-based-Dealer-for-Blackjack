import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tkinter import *

# Load the dataset
data = pd.read_csv('blackjack_data.csv')

#Implement the random forest classifier 
data = data.drop(columns=['player_action'])
X = data.drop('outcome', axis=1)
y = data['outcome']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Define the GUI window
root = Tk()
root.geometry("400x200")
root.title("Blackjack Player Behavior Analyzer")
num_cards_label = Label(root, text="Number of Cards:", font=("Arial", 14))
num_cards_entry = Entry(root, font=("Arial", 14))
player_score_label = Label(root, text="Player Score:", font=("Arial", 14))
player_score_entry = Entry(root, font=("Arial", 14))
dealer_score_label = Label(root, text="Dealer Score:", font=("Arial", 14))
dealer_score_entry = Entry(root, font=("Arial", 14))
dealer_upcard_label = Label(root, text="Dealer Upcard:", font=("Arial", 14))
dealer_upcard_entry = Entry(root, font=("Arial", 14))
output_label = Label(root, text="", font=("Arial", 14))

# Define the analyze function
def analyze():
    # Get the user input
    num_cards = int(num_cards_entry.get())
    player_score = int(player_score_entry.get())
    dealer_score = int(dealer_score_entry.get())
    dealer_upcard = int(dealer_upcard_entry.get())
    
    # Make the prediction
    input_data = [[num_cards, player_score, dealer_score, dealer_upcard]]
    prediction = rf.predict(input_data)
    
    # Display the prediction
    output_label.configure(text="Predicted Behavior: " + prediction[0], font=("Arial", 14))

# Define the layout of the widgets
num_cards_label.grid(row=0, column=0)
num_cards_entry.grid(row=0, column=1)
player_score_label.grid(row=1, column=0)
player_score_entry.grid(row=1, column=1)
dealer_score_label.grid(row=2, column=0)
dealer_score_entry.grid(row=2, column=1)
dealer_upcard_label.grid(row=3, column=0)
dealer_upcard_entry.grid(row=3, column=1)
output_label.grid(row=4, column=0, columnspan=2)
Button(root, text="Analyze", command=analyze, font=("Arial", 14)).grid(row=5, column=0, columnspan=2)

# Start the GUI
root.mainloop()