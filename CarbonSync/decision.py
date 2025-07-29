import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

DEFAULT_DC_SIZE = 1000  # in m²

class  ChainedPredictor:
    global solar_energy, wind_energy, cooling_energy
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

    def predict_monthly(self, region, country, year, dc_size_m2=None):
        dc_size = dc_size_m2 or DEFAULT_DC_SIZE
        results = []

        for month in range(1, 13):
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

            # Calculate energy values
            solar_energy = assd_pred * dc_size * 30  # kWh/month
            wind_energy = 0.5 * 1.225 * (wind_pred**3) * dc_size * 0.4 * 24 * 30 / 1000  # in kWh (approx)
            cooling_energy = dc_size * self.n_factor(temp_pred) * 24 * 30 / 1000  # in kWh

            results.append({
                'Month': month,
                'ASSD(kWh/m²/day)': assd_pred,
                'Temp(C)': temp_pred,
                'SP(kPa)': sp_pred,
                'wind speed(m/s)': wind_pred,
                'SolarEnergy(kWh)': solar_energy,
                'WindEnergy(kWh)': wind_energy,
                'CoolingEnergy(kWh)': cooling_energy
            })

        return pd.DataFrame(results)

    def n_factor(self, temp):
        if temp <= 10:
            return 100
        elif temp <= 15:
            return 130
        elif temp <= 20:
            return 160
        elif temp <= 25:
            return 190
        elif temp <= 30:
            return 220
        elif temp <= 35:
            return 250
        else:
            return 280


if __name__ == '__main__':
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
    df_result = predictor.predict_monthly("Asia", "India", 2025, dc_size_m2=2000)
    print(df_result)
    # Save results to CSV
    df_result.to_csv('monthly_predictions.csv', index=False)