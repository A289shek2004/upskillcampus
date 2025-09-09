📄 README – Turbofan Engine RUL Prediction

📌 Project Overview

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS dataset.
It implements and evaluates multiple machine learning and deep learning models to estimate how many cycles an engine can operate before failure, with the final deployment-ready solution built using Streamlit.

📊 Dataset

Source: NASA C-MAPSS (FD001–FD004 subsets)

Each dataset contains:

* Unit ID – engine identifier
* Cycle – operational cycle
* Operating conditions – 3 settings
* Sensor readings – 26 sensors (s1–s26)

Labels:

* RUL = (Max cycle per engine – Current cycle)
* For test sets, NASA provides ground-truth RUL values.

⚙️ Pipeline Workflow

**Data Preprocessing**

* Loaded raw sensor data from train\_FD001.txt and test\_FD001.txt.
* Normalized and smoothed sensor readings (rolling averages).
* Created target labels for RUL prediction.

**Feature Engineering**

* Time-series windowing for sequence learning.
* Noise reduction techniques applied.

**Model Training & Evaluation**
Implemented and compared:

* Linear Regression
* Random Forest
* LSTM
* BiLSTM
* GRU
* CNN-LSTM
* Attention-LSTM
* Transformer

**Evaluation Metrics**

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² (Coefficient of Determination)
* NASA Scoring Function (penalizes late predictions more than early ones).

📈 Results

### Production Models (Deployment Candidates)

Model	Seq\_len	Units	Dropout	RMSE	MAE	R²	NASA
LSTM	50	128	0.3	12.47	9.18	0.91	8375
GRU	50	128	0.3	11.47	8.44	0.92	6234
LSTM	40	128	0.3	12.59	9.12	0.91	9628
GRU	40	128	0.3	12.29	8.90	0.91	8900

✅ Best Performer: **GRU (Seq\_len=50, Units=128, Dropout=0.3)** → RMSE \~11.47, R² \~0.923, NASA \~6234.

### Research Models (Explored, Not Deployment-Ready)

* BiLSTM → Slightly worse R², overfitting risk.
* CNN-LSTM → Validation decent, test collapse in earlier trials.
* Attention-LSTM → Very high NASA score, unstable.
* Transformer → Inconsistent generalization, needs larger datasets.

🛠️ Project Files

* **turbofan\_rul\_final.py** – Full pipeline (data preprocessing, training, evaluation, visualization).
* **app.py** – Streamlit dashboard for interactive RUL prediction.
* **outputs/** – Saved model results, plots, metrics.
* **data/** – NASA C-MAPSS dataset files (train/test/RUL).
* **models/** – Trained models (checkpoints: best LSTM/GRU).

📌 Key Learnings

* Sequence models (LSTM, GRU) are highly effective for time-series degradation prediction.
* GRU (Seq\_len=50, Units=128) provided the **most stable and accurate results**.
* Traditional models (Linear Regression, Random Forest) serve as interpretable baselines but lack deep feature capture.
* CNN-LSTM and Transformers require careful regularization and larger datasets to perform well.
* Visualizations of Predicted vs Actual RUL are crucial for industry adoption.

🚀 Future Work

* Extend evaluation to all C-MAPSS datasets (FD001–FD004) for generalization.
* Experiment with ensemble deep learning models (stacking LSTM, GRU, CNN).
* Use attention mechanisms for **sensor-level interpretability**.
* Deploy via Streamlit/FastAPI for real-time engine health monitoring.
* Add batch mode predictions for scalability across multiple engines daily.

✅ This project demonstrates a complete predictive maintenance pipeline, from raw sensor data preprocessing to deployment-ready RUL prediction, with reproducible experiments, professional reporting, and a Streamlit-based interactive dashboard.
