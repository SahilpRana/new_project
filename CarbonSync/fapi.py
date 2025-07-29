from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


DEFAULT_AREA = 1000.0       # m² for data center roof
DEFAULT_ROTOR_AREA = 200.0  # m² for wind turbine
DEFAULT_EFFICIENCY = 20     # % solar panel efficiency

N_FACTORS = {
    0: 0.1, 5: 0.2, 10: 0.4, 15: 0.6, 20: 0.75,
    25: 0.9, 30: 1.0, 35: 1.1, 40: 1.25
}

def get_n_factor(temp):
    keys = sorted(N_FACTORS.keys())
    return np.interp(temp, keys, [N_FACTORS[k] for k in keys])

def calculate_solar_energy(assd, area, efficiency):
    daily_energy = assd * area * (efficiency / 100.0)  # kWh/day
    return daily_energy * 30  # monthly

def calculate_wind_energy(ws, rotor_area):
    rho = 1.225  # air density kg/m³
    cp = 0.4     # power coefficient
    return 0.5 * rho * (ws ** 3) * rotor_area * cp * 24 * 30 / 1000  # kWh/month

def calculate_cooling_energy(temp, area):
    n_factor = get_n_factor(temp)
    return area * n_factor * 24 * 30 / 1000  # kWh/month

class ChainedPredictor:
    def __init__(self, model_paths):
        self.model_assd = load_model(model_paths['assd'])
        self.model_temp = joblib.load(model_paths['temp'])
        self.model_sp   = joblib.load(model_paths['sp'])
        self.model_wind = joblib.load(model_paths['wind'])

        # Load preprocessors
        self.encoder1 = joblib.load(model_paths['encoder1'])
        self.scaler1  = joblib.load(model_paths['scaler1'])
        self.encoder2 = joblib.load(model_paths['encoder2'])
        self.scaler2  = joblib.load(model_paths['scaler2'])
        self.encoder3 = joblib.load(model_paths['encoder3'])
        self.scaler3  = joblib.load(model_paths['scaler3'])
        self.encoder4 = joblib.load(model_paths['encoder4'])
        self.scaler4  = joblib.load(model_paths['scaler4'])

        # Feature names
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

app = FastAPI(title="Chained Predictor API with Energy Calculations")

class PredictRequest(BaseModel):
    region: str
    country: str
    year: int
    month: int
    area: float = DEFAULT_AREA
    rotor_area: float = DEFAULT_ROTOR_AREA
    efficiency: float = DEFAULT_EFFICIENCY

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

@app.post("/predict")
async def predict(data: PredictRequest):
    # Climate predictions
    result = predictor.predict(data.region, data.country, data.year, data.month)

    # Energy calculations
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
