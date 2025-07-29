import pandas as pd
import numpy as np
from cp import ChainedPredictor
from datetime import datetime
from tqdm import tqdm 

# Constants
YEARS = 15
MONTHS = YEARS * 12
DEFAULT_SIZE_MW = 5  
TOP_N = 5
N_FACTORS = {
    0: 0.1, 5: 0.2, 10: 0.4, 15: 0.6, 20: 0.75,
    25: 0.9, 30: 1.0, 35: 1.1, 40: 1.25
}

# ---------- Helper functions ----------
def get_n_factor(temp):
    keys = sorted(N_FACTORS.keys())
    return np.interp(temp, keys, [N_FACTORS[k] for k in keys])

def solar_energy(assd, dc_size):
    return assd * dc_size * 30  # kWh/month

def wind_energy(wind_speed, dc_size):
    return 0.5 * 1.225 * (wind_speed**3) * dc_size * 0.4 * 24 * 30 / 1000  # kWh/month

def cooling_energy(temp, dc_size):
    n_factor = get_n_factor(temp)
    return dc_size * n_factor * 24 * 30 / 1000  # kWh/month

# ---------- Load data ----------
df = pd.read_csv("C:\\Users\\sahil\\new_project\\new_project\\CarbonSync\\Datagathering\\Data.csv")
df['Region'] = df['Region'].str.strip()
df['Country'] = df['Country'].str.strip()

# ---------- User input ----------
region = input("Enter region (e.g., Europe, Asia): ").strip()

# Validate region
countries = df[df['Region'] == region]['Country'].unique()
if len(countries) == 0:
    print(f"No countries found for region: {region}")
    exit()

# Default data center size
size_mw = DEFAULT_SIZE_MW

# Generate timestamps
now = datetime.now()
timestamps = pd.date_range(start=now, periods=MONTHS, freq='MS').strftime("%Y-%m").tolist()

# Initialize predictor
model_paths = {
        'assd':     r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\assd_lstm_model.keras',
        'temp':     r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\temp.pkl',
        'sp':       r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\SP.pkl',
        'wind':     r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\wind_speed.pkl',
        'encoder1': r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\encoder1.pkl',
        'scaler1':  r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\scaler1.pkl',
        'encoder2': r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\encoder2.pkl',
        'scaler2':  r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\scaler2.pkl',
        'encoder3': r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\encoder3.pkl',
        'scaler3':  r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\scaler3.pkl',
        'encoder4': r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\encoder4.pkl',
        'scaler4':  r'C:\Users\sahil\new_project\new_project\CarbonSync\Models\scaler4.pkl'
}
predictor = ChainedPredictor(model_paths)

results = []

for country in tqdm(countries, desc="Processing countries"):
    all_assd, all_temp, all_ws = [], [], []

    for ts in tqdm(timestamps, desc=f"Predicting {country}", leave=False):
        try:
            year, month = map(int, ts.split('-'))
            pred = predictor.predict(region, country, year, month)
            all_assd.append(pred['ASSD(kWh/m²/day)'])
            all_temp.append(pred['Temp(C)'])
            all_ws.append(pred['wind speed(m/s)'])
        except Exception as e:
            print(f"⚠️ Error in {country} for {ts}: {e}")
            continue

    if not all_assd:
        continue

    all_assd = np.array(all_assd)
    all_temp = np.array(all_temp)
    all_ws = np.array(all_ws)

    # Energy calculations
    solar_total = np.sum([solar_energy(a, size_mw) for a in all_assd])
    wind_total  = np.sum([wind_energy(w, size_mw) for w in all_ws])
    cool_total  = np.sum([cooling_energy(t, size_mw) for t in all_temp])
    score = solar_total + wind_total - cool_total

    results.append({
        "country": country,
        "solar_energy": round(solar_total, 2),
        "wind_energy": round(wind_total, 2),
        "cooling_energy": round(cool_total, 2),
        "score": round(score, 2)
    })

# ---------- Ranking ----------
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="score", ascending=False).reset_index(drop=True)
df_result["rank"] = df_result.index + 1
df_result = df_result[["rank", "country", "solar_energy", "wind_energy", "cooling_energy", "score"]]

# Show top N
top_n = df_result.head(TOP_N)
print(f"\n🔍 Top {TOP_N} Countries for Region: {region} ({YEARS} years forecast)\n")
print(top_n.to_string(index=False))

best = top_n.iloc[0]
print(f"\n✅ Suggested best country: **{best['country']}** (Rank 1)")

# Optional save
top_n.to_csv(f"top_{TOP_N}_{region}_ranking.csv", index=False)
