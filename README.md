# Hydraulic Systems Condition Monitoring 🛠️

## 📖 Overview
This project presents an end-to-end AI-based solution for **Condition Monitoring and Predictive Maintenance** of hydraulic systems. The primary goal is to shift from traditional time-based maintenance to **Predictive Maintenance (PdM)**, effectively reducing **Unplanned Downtime** and extending the operational lifespan of critical hydraulic components.

The system utilizes **Random Forest** machine learning models to analyze multi-sensor data (pressure, temperature, flow, and vibration) to diagnose the health status of components such as valves, pumps, and coolers.



## 🚀 Key Features
* **Failure Diagnosis:** Real-time health assessment of critical hydraulic components.
* **Model Insights:** Transparent decision-making through **SHAP (SHapley Additive exPlanations)** to understand model predictions.
* **Interactive Dashboard:** An intuitive **Streamlit** dashboard for visualizing sensor correlations and model diagnostics.

## 🛠️ Technical Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn (Random Forest)
* **Explainability:** SHAP
* **Visualization:** Streamlit, Pandas, Matplotlib, Plotly
* **Environment:** Virtual Environment (venv)

## 📊 Dataset
The original dataset used in this project is part of the "Condition Monitoring of Hydraulic Systems" dataset. Due to the large file size, the data is not included in this repository.

To run the project, please follow these steps:
1. Download the dataset from the official source: [(https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)]
2. Extract the downloaded files.
3. Create a folder named `data` in the project root directory.
4. Place the `.txt` files inside the `data` folder.

If you only want to test the dashboard with a small sample, I have included a small subset of the data in the `sample_data/` folder.

## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/hydraulic-monitoring.git](https://github.com/your-username/hydraulic-monitoring.git)
   cd hydraulic-monitoring
2. **Setup virtual environment:**

   ```Bash

   python -m venv .venv
   .\.venv\Scripts\activate  # Windows

3. **Install dependencies:**

   ```Bash

   pip install -r requirements.txt
4. **Run the dashboard:**

   ```Bash

   streamlit run main.py

📈 Results

The model achieves over 95% accuracy in classifying component failure modes, providing actionable engineering insights into the most influential sensors for each specific fault condition.

👤 Author
[Mohamedin Khalfalla]

LinkedIn: [www.linkedin.com/in/mohamedin-khalfallah-3702b4345]

GitHub: [https://github.com/khalifakhalaf01]
