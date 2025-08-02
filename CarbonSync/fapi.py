from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json
import gdown

# ======================
# Default Parameters
# ======================
DEFAULT_AREA = 1000.0       # m²
DEFAULT_ROTOR_AREA = 200.0  # m²
DEFAULT_EFFICIENCY = 20     # %
YEARS = 15
MONTHS = YEARS * 12
TOP_N = 5

# Cooling factor mapping
N_FACTORS = {
    0: 0.1, 5: 0.2, 10: 0.4, 15: 0.6, 20: 0.75,
    25: 0.9, 30: 1.0, 35: 1.1, 40: 1.25
}

# ======================
# Helper Functions
# ======================
def get_n_factor(temp):
    keys = sorted(N_FACTORS.keys())
    return np.interp(temp, keys, [N_FACTORS[k] for k in keys])

def calculate_solar_energy(assd, area, efficiency):
    return assd * area * (efficiency / 100.0) * 30

def calculate_wind_energy(ws, rotor_area):
    rho = 1.225
    cp = 0.4
    return 0.5 * rho * (ws ** 3) * rotor_area * cp * 24 * 30 / 1000

def calculate_cooling_energy(temp, area):
    n_factor = get_n_factor(temp)
    return area * n_factor * 24 * 30 / 1000

# ======================
# Download models from Drive (Temp, SP, Wind only)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

DRIVE_MODELS = {
    "assd_lstm_model.keras": "1xwT_E3ICZyvMpY4dyzcLSfjHNNo7lZWw",
    "temp.pkl": "1pJEzxYzfYL8nsZInQoFEgYCTQrnGtrP5",
    "SP.pkl": "17QdJ5wr0gj1ebhCpJ-J6AM2QCJb_yZ00",
    "wind_speed.pkl": "1btBovQXlj307sCbOU7jCLEph_VF1AHSA",
    "encoder1.pkl": "18cDvazi0YMz2VEW6HRiqPms-HKCok5OA",
    "scaler1.pkl": "1YvvAHEbZt3QGuwwSyR4Q-QbnWKiL0B4P",
    "encoder2.pkl": "1Tnb7EFR48SydkWplwA-yFRgapCpydiuM",
    "scaler2.pkl": "1_vWV_6KssIvjAqq609UutWy2xc8Zp4n1",
    "encoder3.pkl": "101t-FuI5nQHvNIL-gpDs7CvRFuSCdDtE",
    "scaler3.pkl": "1oZrOq2VGXQybUGN6leFhzNEkgxMf1bTn",
    "encoder4.pkl": "1YpfTj0NKOu9CCpo9k8l-hnAk7Dj_MFo-",
    "scaler4.pkl": "1UibquSt9h5hW-S6nUBNtcc044HaKCjW1"
}

def download_drive_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, file_id in DRIVE_MODELS.items():
        file_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

# Download before loading
download_drive_models()

# ======================
# Predictor Class
# ======================
class ChainedPredictor:
    def __init__(self, model_paths):
        self.model_assd = load_model(model_paths['assd'])
        self.model_temp = joblib.load(model_paths['temp'])
        self.model_sp   = joblib.load(model_paths['sp'])
        self.model_wind = joblib.load(model_paths['wind'])
        self.encoder1 = joblib.load(model_paths['encoder1'])
        self.scaler1  = joblib.load(model_paths['scaler1'])
        self.encoder2 = joblib.load(model_paths['encoder2'])
        self.scaler2  = joblib.load(model_paths['scaler2'])
        self.encoder3 = joblib.load(model_paths['encoder3'])
        self.scaler3  = joblib.load(model_paths['scaler3'])
        self.encoder4 = joblib.load(model_paths['encoder4'])
        self.scaler4  = joblib.load(model_paths['scaler4'])
        self.num_cols1 = list(self.scaler1.feature_names_in_)
        self.num_cols2 = list(self.scaler2.feature_names_in_)
        self.num_cols3 = list(self.scaler3.feature_names_in_)
        self.num_cols4 = list(self.scaler4.feature_names_in_)

    def _encode_cat(self, encoder, region, country):
        df_cat = pd.DataFrame([{'Region': region, 'Country': country}])
        return encoder.transform(df_cat)

    def _scale_num(self, scaler, cols, df_num):
        df_num = df_num.reindex(columns=cols, fill_value=0)
        return scaler.transform(df_num)

    def preprocess_lstm_input(self, region, country, year, month):
        df_num = pd.DataFrame([{'year ': year, 'month': month}])
        num_scaled = self._scale_num(self.scaler1, self.num_cols1, df_num)
        cat_enc = self._encode_cat(self.encoder1, region, country)
        return np.hstack([num_scaled, cat_enc]).reshape(1, 1, -1).astype(np.float32)

    def preprocess_features2(self, region, country, year, month, assd):
        df_num = pd.DataFrame([{
            'year ': year,
            'month': month,
            'ASSD(kWh/m²/day)': assd
        }])
        num_scaled = self._scale_num(self.scaler2, self.num_cols2, df_num)
        cat_enc = self._encode_cat(self.encoder2, region, country)
        return np.hstack([num_scaled, cat_enc])

    def preprocess_features3(self, region, country, year, month, assd, temp):
        df_num = pd.DataFrame([{
            'year ': year,
            'month': month,
            'ASSD(kWh/m²/day)': assd,
            'Temp(C)': temp
        }])
        num_scaled = self._scale_num(self.scaler3, self.num_cols3, df_num)
        cat_enc = self._encode_cat(self.encoder3, region, country)
        return np.hstack([num_scaled, cat_enc])

    def preprocess_features4(self, region, country, year, month, assd, temp, sp):
        df_num = pd.DataFrame([{
            'year ': year,
            'month': month,
            'ASSD(kWh/m²/day)': assd,
            'Temp(C)': temp,
            'SP(kPa)': sp
        }])
        num_scaled = self._scale_num(self.scaler4, self.num_cols4, df_num)
        cat_enc = self._encode_cat(self.encoder4, region, country)
        return np.hstack([num_scaled, cat_enc])

    def predict(self, region, country, year, month):
        assd_pred = float(self.model_assd.predict(
            self.preprocess_lstm_input(region, country, year, month), verbose=0
        )[0, 0])

        temp_pred = float(self.model_temp.predict(
            self.preprocess_features2(region, country, year, month, assd_pred).reshape(1, -1)
        )[0])

        sp_pred = float(self.model_sp.predict(
            self.preprocess_features3(region, country, year, month, assd_pred, temp_pred).reshape(1, -1)
        )[0])

        wind_pred = float(self.model_wind.predict(
            self.preprocess_features4(region, country, year, month, assd_pred, temp_pred, sp_pred).reshape(1, -1)
        )[0])

        return {
            'ASSD(kWh/m²/day)': assd_pred,
            'Temp(C)': temp_pred,
            'SP(kPa)': sp_pred,
            'wind speed(m/s)': wind_pred
        }

# ======================
# FastAPI Setup
# ======================
app = FastAPI(title="Chained Predictor API with Energy Calculations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is running. Use POST /predict or /rank"}

# ======================
# Schemas
# ======================
class PredictRequest(BaseModel):
    region: str
    country: str
    year: int
    month: int
    area: float = DEFAULT_AREA
    rotor_area: float = DEFAULT_ROTOR_AREA
    efficiency: float = DEFAULT_EFFICIENCY

class RankRequest(BaseModel):
    region: str

# ======================
# Load models
# ======================
model_paths = {
    'assd':     os.path.join(MODELS_DIR, "assd_lstm_model.keras"),
    'temp':     os.path.join(MODELS_DIR, "temp.pkl"),
    'sp':       os.path.join(MODELS_DIR, "SP.pkl"),
    'wind':     os.path.join(MODELS_DIR, "wind_speed.pkl"),
    'encoder1': os.path.join(MODELS_DIR, "encoder1.pkl"),
    'scaler1':  os.path.join(MODELS_DIR, "scaler1.pkl"),
    'encoder2': os.path.join(MODELS_DIR, "encoder2.pkl"),
    'scaler2':  os.path.join(MODELS_DIR, "scaler2.pkl"),
    'encoder3': os.path.join(MODELS_DIR, "encoder3.pkl"),
    'scaler3':  os.path.join(MODELS_DIR, "scaler3.pkl"),
    'encoder4': os.path.join(MODELS_DIR, "encoder4.pkl"),
    'scaler4':  os.path.join(MODELS_DIR, "scaler4.pkl")
}
predictor = ChainedPredictor(model_paths)

# ======================
# Endpoints
# ======================

# Single prediction
@app.post("/predict")
async def predict(data: PredictRequest):
    result = predictor.predict(data.region, data.country, data.year, data.month)
    solar_energy = calculate_solar_energy(result['ASSD(kWh/m²/day)'], data.area, data.efficiency)
    wind_energy = calculate_wind_energy(result['wind speed(m/s)'], data.rotor_area)
    cooling_energy = calculate_cooling_energy(result['Temp(C)'], data.area)
    net_energy = solar_energy + wind_energy - cooling_energy

    return {
        "predictions": result,
        "energy_calculations": {
            "Solar Energy (kWh/month)": round(solar_energy, 2),
            "Wind Energy (kWh/month)": round(wind_energy, 2),
            "Cooling Energy (kWh/month)": round(cooling_energy, 2),
            "Net Energy Balance (kWh/month)": round(net_energy, 2)
        }
    }

# Ranking endpoint
RANKINGS_PATH = os.path.join(BASE_DIR, "Datagathering", "RegionalRankings.json")

@app.post("/rank")
async def rank_region(data: RankRequest):
    # Load existing RegionalRankings.json
    if os.path.exists(RANKINGS_PATH):
        with open(RANKINGS_PATH, "r") as f:
            try:
                regional_rankings = json.load(f)
            except json.JSONDecodeError:
                regional_rankings = {"data": []}
    else:
        regional_rankings = {"data": []}

    # 1. Check if region already exists
    existing = next((r for r in regional_rankings["data"] if r["region"].lower() == data.region.lower()), None)
    if existing:
        return existing  # Return cached data immediately

    # 2. Compute rankings if not cached
    df = pd.read_csv(os.path.join(BASE_DIR, "Datagathering", "Data.csv"))
    df['Region'] = df['Region'].str.strip()
    df['Country'] = df['Country'].str.strip()

    countries = df[df['Region'].str.lower() == data.region.lower()]['Country'].unique()
    if len(countries) == 0:
        return {"error": f"No countries found for region: {data.region}"}

    # Generate timestamps
    now = datetime.now()
    timestamps = pd.date_range(start=now, periods=MONTHS, freq='MS').strftime("%Y-%m").tolist()

    results = []

    # Loop through countries
    for country in countries:
        all_assd, all_temp, all_ws = [], [], []
        for ts in timestamps:
            year, month = map(int, ts.split('-'))
            pred = predictor.predict(data.region, country, year, month)
            all_assd.append(pred['ASSD(kWh/m²/day)'])
            all_temp.append(pred['Temp(C)'])
            all_ws.append(pred['wind speed(m/s)'])

        # Energy totals
        solar_total = np.sum([calculate_solar_energy(a, DEFAULT_AREA, DEFAULT_EFFICIENCY) for a in all_assd])
        wind_total  = np.sum([calculate_wind_energy(w, DEFAULT_ROTOR_AREA) for w in all_ws])
        cool_total  = np.sum([calculate_cooling_energy(t, DEFAULT_AREA) for t in all_temp])
        score = solar_total + wind_total - cool_total

        results.append({
            "country": country,
            "solar_energy": round(solar_total, 2),
            "wind_energy": round(wind_total, 2),
            "cooling_energy": round(cool_total, 2),
            "score": round(score, 2)
        })

    # Rank results
    df_result = pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
    df_result["rank"] = df_result.index + 1
    top_n = df_result.head(TOP_N).to_dict(orient="records")

    result_data = {
        "region": data.region,
        "years_forecast": YEARS,
        "top_countries": top_n,
        "best_country": top_n[0]["country"]
    }

    # 3. Save new result to RegionalRankings.json
    regional_rankings["data"].append(result_data)
    with open(RANKINGS_PATH, "w") as f:
        json.dump(regional_rankings, f, indent=2)

    return result_data
