# **CarbonSync – Smarter Energy, Greener Data Centers**

CarbonSync is an AI-powered platform designed to forecast energy consumption and carbon emissions in data centers, helping operators make proactive decisions to reduce costs and environmental impact. Using a chained neural network approach, CarbonSync predicts environmental factors such as solar radiation (ASSD), temperature, surface pressure, wind speed, and precipitation, feeding these into a recommendation engine that optimizes cooling and energy efficiency in real time. The platform is built with **FastAPI** for backend services, **TensorFlow/Keras** and **scikit-learn** for machine learning models, and **Pandas/Numpy** for data handling. It provides both an API endpoint for integrations and an interactive dashboard for visualization. While CarbonSync is functional, it currently depends on historical datasets and may miss sudden anomalies; real-time sensor integration and simplified deployment are in active development. Future plans include renewable energy forecasting, anomaly detection for cooling failures, and a tokenized carbon credit marketplace — laying the groundwork for scalable, sustainable digital infrastructure.

---

## **Installation**

```bash
git clone <repo-link>
cd CarbonSync
pip install -r requirements.txt
python fapi.py
```

---

## **Usage**

* Access predictions via API endpoints exposed by FastAPI.
* Integrate with dashboards or external systems for monitoring.
* Use ngrok (or similar) to expose local server publicly if needed.

---

## **Tech Stack**

* **Backend:** FastAPI, Python
* **ML Models:** TensorFlow/Keras, scikit-learn
* **Data Handling:** Pandas, Numpy
* **Deployment:** ngrok / Hugging Face Spaces

---

## **Roadmap**

* Add renewable energy forecasting
* Implement anomaly detection for cooling failures
* Build carbon credit trading module
* Enhance real-time sensor integrations and scaling

---

## **Contributors**

* @sahil\_rana – Sahil Rana
