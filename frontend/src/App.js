import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [region, setRegion] = useState('');
  const [country, setCountry] = useState('');
  const [year, setYear] = useState('');
  const [month, setMonth] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:8000/predict', {
        region,
        country,
        year: parseInt(year),
        month: parseInt(month)
      });
      setPrediction(res.data);
    } catch (err) {
      alert('Prediction API call failed. Check backend.');
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>CarbonSync Energy Predictor</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Region: </label>
          <input value={region} onChange={e => setRegion(e.target.value)} />
        </div>
        <div>
          <label>Country: </label>
          <input value={country} onChange={e => setCountry(e.target.value)} />
        </div>
        <div>
          <label>Year: </label>
          <input value={year} onChange={e => setYear(e.target.value)} />
        </div>
        <div>
          <label>Month: </label>
          <input value={month} onChange={e => setMonth(e.target.value)} />
        </div>
        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Predicted Results</h2>
          <ul>
            <li><strong>ASSD:</strong> {prediction.assd}</li>
            <li><strong>Temperature (°C):</strong> {prediction.temp}</li>
            <li><strong>Surface Pressure (kPa):</strong> {prediction.sp}</li>
            <li><strong>Wind Speed (m/s):</strong> {prediction.wind_speed}</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;