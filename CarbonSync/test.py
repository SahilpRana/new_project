import os

path = r"C:\Users\sahil\new_project\CarbonSync\Models\assd_lstm_model.keras"
print("Exists:", os.path.exists(path))
print("Size (bytes):", os.path.getsize(path))

# Quick check if it's actually a directory
print("Is dir?", os.path.isdir(path))
