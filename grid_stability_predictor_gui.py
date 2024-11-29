# Tkinter GUI for file upload and predictions
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import joblib  # For saving/loading the fitted scaler

# Load the pre-trained model
classifier = Sequential()
classifier.add(Dense(units=24, activation='relu', input_dim=12))
classifier.add(Dense(units=24, activation='relu'))
classifier.add(Dense(units=12, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the model's saved weights
classifier.load_weights(r'C:\sr\model.weights.h5')

# Load the previously fitted StandardScaler
scaler = joblib.load(r'C:\sr\scaler.pkl')  # Assuming you saved the scaler during training


# Function to generate explanations based on predictions
def get_explanation(prediction, features):
    if prediction == 0:  # Unstable
        explanation = "The grid is unstable due to the following factors:\n"

        # Analyze tau values
        if features['tau1'] > 5:  # Supplier reaction time
            explanation += "- Supplier reaction time (tau1) is too high (greater than 5), causing delays in power adjustment.\n"
        if features['tau2'] > 5:  # Consumer 1 reaction time
            explanation += "- Consumer 1 reaction time (tau2) is too high (greater than 5), leading to delayed response in power consumption.\n"
        if features['tau3'] > 5:  # Consumer 2 reaction time
            explanation += "- Consumer 2 reaction time (tau3) is too high (greater than 5), causing instability due to slow adjustments.\n"
        if features['tau4'] > 5:  # Consumer 3 reaction time
            explanation += "- Consumer 3 reaction time (tau4) is too high (greater than 5), contributing to the overall instability.\n"

        # Analyze power values
        total_consumed = features['p2'] + features['p3'] + features['p4']
        if total_consumed > features['p1']:  # Imbalance between supply and demand
            explanation += f"- Total power consumed ({total_consumed}) exceeds power produced by the supplier (p1 = {features['p1']}), leading to instability.\n"

        if features['p2'] < -2 or features['p3'] < -2 or features['p4'] < -2:  # Example of extreme consumption
            explanation += "- One or more consumers are consuming excessive power, contributing to instability.\n"

        # Analyze price elasticity coefficients
        if features['g1'] < 0.1:  # Supplier price elasticity
            explanation += "- Supplier price elasticity (g1) is too low, indicating an inadequate response to market changes.\n"
        if features['g2'] < 0.1:  # Consumer 1 price elasticity
            explanation += "- Consumer 1 price elasticity (g2) is too low, indicating a lack of responsiveness to price signals.\n"
        if features['g3'] < 0.1:  # Consumer 2 price elasticity
            explanation += "- Consumer 2 price elasticity (g3) is too low, limiting the effectiveness of price adjustments.\n"
        if features['g4'] < 0.1:  # Consumer 3 price elasticity
            explanation += "- Consumer 3 price elasticity (g4) is too low, which may hinder stability through insufficient demand response.\n"

        return explanation
    else:
        return "The grid is stable."


# Function to upload a CSV file and predict grid stability
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            # Read and preprocess the uploaded CSV file
            grid_data = pd.read_csv(file_path)
            feature_columns = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']

            if set(feature_columns).issubset(grid_data.columns):
                # Scale the data using the previously fitted scaler
                grid_data_scaled = scaler.transform(grid_data[feature_columns])

                # Make predictions using the model
                predictions = classifier.predict(grid_data_scaled)
                predictions_binary = (predictions > 0.5).astype(int)  # Thresholding for binary classification

                # Map binary predictions to "Stable" or "Unstable"
                predictions_labels = ['Stable' if pred == 1 else 'Unstable' for pred in predictions_binary.flatten()]

                # Prepare results with explanations
                results = []
                for i in range(len(predictions_labels)):
                    explanation = get_explanation(predictions_binary[i], grid_data.iloc[i].to_dict())
                    result_line = f"Prediction: {predictions_labels[i]}\nExplanation:\n{explanation}\n"
                    results.append(result_line)

                # Display predictions and explanations in the text box
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, ''.join(results))
            else:
                messagebox.showerror("Error", "CSV file does not have the correct feature columns.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Initialize Tkinter window
root = tk.Tk()
root.title("Grid Stability Predictor")

# Create UI elements
label = tk.Label(root, text="Upload a CSV file to predict grid stability:")
label.pack(pady=10)

upload_button = tk.Button(root, text="Upload CSV", command=upload_file)
upload_button.pack(pady=10)

# Text box to display results
result_text = ScrolledText(root, width=80, height=20)
result_text.pack(pady=10)

# Run the Tkinter loop
root.mainloop()
