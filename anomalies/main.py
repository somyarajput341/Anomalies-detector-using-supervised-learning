import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx")])

print("File selected:", file_path)

if not file_path:
    raise Exception("No file selected.")

df = pd.read_excel(file_path)

numeric_df = df.select_dtypes(include=[np.number])
print("Numeric columns detected:", numeric_df.columns.tolist())

if numeric_df.empty:
    raise Exception("No numeric columns found.")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

clf = IsolationForest(contamination=0.15, random_state=42)
clf.fit(scaled_data)
preds = clf.predict(scaled_data)

anomaly_idx = np.where(preds == -1)[0]

print("Anomalies count:", len(anomaly_idx))
print("Anomaly indices:", anomaly_idx)

for col in numeric_df.columns:
    plt.figure(figsize=(10,5))
    plt.plot(numeric_df.index, numeric_df[col], label="Normal Data")
    plt.scatter(
        anomaly_idx, 
        numeric_df[col].iloc[anomaly_idx], 
        color='red', 
        s=80, 
        marker='x', 
        label="Anomaly"
    )
    plt.title(f"Anomaly Detection: {col}")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

input("Press Enter to exit...")
 