import os
import streamlit as st
import numpy as np
from cp import ChainedPredictor

# --------------------------
# Assumptions / Constants
# --------------------------
# Default Data Center Specs
DEFAULT_AREA = 1000.0       # m² for data center roof (solar panels)
DEFAULT_ROTOR_AREA = 200.0  # m² rotor swept area for wind turbine
DEFAULT_EFFICIENCY = 20     # % solar panel efficiency

# Cooling factor (n-factor) based on temperature °C
N_FACTORS = {
    0: 0.1, 5: 0.2, 10: 0.4, 15: 0.6, 20: 0.75,
    25: 0.9, 30: 1.0, 35: 1.1, 40: 1.25
}

# --------------------------
# Helper Functions
# --------------------------
def get_n_factor(temp):
    """Interpolate cooling factor based on temperature (°C)."""
    keys = sorted(N_FACTORS.keys())
    return np.interp(temp, keys, [N_FACTORS[k] for k in keys])

def calculate_solar_energy(assd, area, efficiency):
    """Monthly solar energy (kWh) from ASSD."""
    daily_energy = assd * area * (efficiency / 100.0)  # kWh/day
    return daily_energy * 30  # monthly approx

def calculate_wind_energy(ws, rotor_area):
    """Monthly wind energy (kWh) simplified formula."""
    rho = 1.225  # kg/m³
    cp = 0.4     # power coefficient
    return 0.5 * rho * (ws ** 3) * rotor_area * cp * 24 * 30 / 1000  # kWh/month

def calculate_cooling_energy(temp, area):
    """Monthly cooling energy (kWh) using n-factor."""
    n_factor = get_n_factor(temp)
    return area * n_factor * 24 * 30 / 1000

# --------------------------
# Model Setup
# --------------------------
BASE_PATH = os.path.dirname(__file__)
model_paths = {
    'assd':     os.path.join(BASE_PATH, 'Models', 'assd_lstm_model.keras'),
    'temp':     os.path.join(BASE_PATH, 'Models', 'temp.pkl'),
    'sp':       os.path.join(BASE_PATH, 'Models', 'SP.pkl'),
    'wind':     os.path.join(BASE_PATH, 'Models', 'wind_speed.pkl'),
    'encoder1': os.path.join(BASE_PATH, 'Models', 'encoder1.pkl'),
    'scaler1':  os.path.join(BASE_PATH, 'Models', 'scaler1.pkl'),
    'encoder2': os.path.join(BASE_PATH, 'Models', 'encoder2.pkl'),
    'scaler2':  os.path.join(BASE_PATH, 'Models', 'scaler2.pkl'),
    'encoder3': os.path.join(BASE_PATH, 'Models', 'encoder3.pkl'),
    'scaler3':  os.path.join(BASE_PATH, 'Models', 'scaler3.pkl'),
    'encoder4': os.path.join(BASE_PATH, 'Models', 'encoder4.pkl'),
    'scaler4':  os.path.join(BASE_PATH, 'Models', 'scaler4.pkl')
}

predictor = ChainedPredictor(model_paths)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌿 CarbonSync: Monthly Energy Forecast")

# --- Show Assumptions ---
with st.expander("Assumptions used in calculations"):
    st.markdown(f""" - **Default Data Center Surface Area:** {DEFAULT_AREA} m²  
    - **Default Wind Rotor Area:** {DEFAULT_ROTOR_AREA} m²  
    - **Default Solar Panel Efficiency:** {DEFAULT_EFFICIENCY}%  
    - **Cooling Factor (n-factor):** Varies with temperature (0°C = 0.1, 40°C = 1.25)  
    - **Energy Units:** All energy values are in kWh (monthly estimates)  
    - **Geographic Coverage:** Currently based on the geographic center points of 134 countries.  
    - **Scalability:** Planned expansion to finer granularity, such as states or sub-regions, for more localized predictions.  
    - **Future Enhancements:** Integration of real-time regional data for higher accuracy and adaptive energy modeling.
    """)

# --- Inputs ---
region = st.selectbox("Select Region", ["Asia"])    # Later connect to Data.csv
country = st.selectbox("Select Country", ["India"])
year = st.number_input("Year", min_value=2020, max_value=2035, value=2025)
month = st.slider("Month", 1, 12, 6)

# Allow overriding defaults
area = st.number_input("Data Center Surface Area (m²)", min_value=100.0, value=DEFAULT_AREA)
efficiency = st.slider("Solar Panel Efficiency (%)", 10, 40, DEFAULT_EFFICIENCY)
rotor_area = st.number_input("Wind Turbine Rotor Area (m²)", min_value=10.0, value=DEFAULT_ROTOR_AREA)

# --------------------------
# Prediction Button
# --------------------------
if st.button("Predict"):
    # Predict climate data for selected month
    result = predictor.predict(region, country, year, month)

    assd = result['ASSD(kWh/m²/day)']
    temp = result['Temp(C)']
    ws = result['wind speed(m/s)']

    # Energy calculations
    solar_energy = calculate_solar_energy(assd, area, efficiency)
    wind_energy = calculate_wind_energy(ws, rotor_area)
    cooling_energy = calculate_cooling_energy(temp, area)
    net_energy = solar_energy + wind_energy - cooling_energy

    # --- Output ---
    st.success(f"Prediction complete for {country} ({month}/{year})!")

    st.write("### Climate Predictions")
    st.json(result)

    st.write("### Energy Calculations (kWh for the month)")
    st.metric("Solar Energy", f"{solar_energy:,.2f}")
    st.metric("Wind Energy", f"{wind_energy:,.2f}")
    st.metric("Cooling Energy", f"{cooling_energy:,.2f}")
    st.metric("Net Energy Balance", f"{net_energy:,.2f}")