📌 Turbofan Engine Remaining Useful Life (RUL) Prediction


A predictive maintenance project using turbofan engine datasets to estimate the Remaining Useful Life (RUL) of aircraft engines.

📖 Introduction

Aircraft engines undergo significant wear and tear during operation. Unexpected engine failure can lead to costly downtime or even catastrophic accidents. Predictive maintenance techniques help avoid these issues by estimating the Remaining Useful Life (RUL) of engines, enabling timely servicing before failure.

This project uses the NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS) to build machine learning models that predict RUL from multivariate time-series sensor data.

📂 Repository Structure
<img width="587" height="312" alt="image" src="https://github.com/user-attachments/assets/fb2a59d5-4863-4616-ade1-894fd6d35ca9" />

✨ Features

📊 Data Handling: Prepares turbofan sensor data from NASA’s C-MAPSS dataset.

🧠 Machine Learning Models: Implements regression and survival models for RUL prediction.

🔄 Time-Series Processing: Extracts engine run-to-failure cycles.

📈 Evaluation Metrics: Calculates RMSE, MAE, and scoring functions.

🚀 Deployment Ready: Simple app.py interface for predictions.

📊 Dataset

The project uses the C-MAPSS Turbofan Engine Degradation Simulation Data.

Training Data (train_FD00X.txt)
Contains multivariate time-series data for multiple engines until failure. Each row = one cycle of an engine.

Testing Data (test_FD00X.txt)
Contains partial histories of engines (not run to failure).

RUL Labels (RUL_FD00X.txt)
Provides the ground truth Remaining Useful Life for each engine in the test set.

Data Format (per row):
id  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1 ... sensor_21


id: Engine unit number

cycle: Time cycle (operational history)

op_setting_x: Operational settings (3)

sensor_x: 21 recorded sensors

⚙️ Getting Started
1. Clone the Repository
git clone https://github.com/A289shek2004/upskillcampus.git

cd upskillcampus

2. Install Dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, typical dependencies are:

pip install numpy pandas scikit-learn matplotlib seaborn

3. Run Training & Evaluation
python turbofan_rul_final.py

4. Launch Application
python app.py


If web-based → navigate to http://127.0.0.1:5000/

If CLI → follow prompts in the terminal

🛠️ Methodology

Data Preprocessing

Handle missing values

Normalize sensor readings

Feature engineering (sliding window, trend features)

Model Development

Regression models: Linear Regression, Random Forest, Gradient Boosting

Neural models (optional extension): LSTMs, RNNs

Evaluation

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

NASA scoring function (penalizes late/early predictions)

Deployment

Save models in models/

Predict on unseen test data and generate outputs in outputs/

📈 Results (Example)

Dataset	Model	RMSE	MAE	Score

FD001	Random Forest	22.5	18.1	550

FD002	Gradient Boosting	28.7	22.4	640

FD003	Linear Regression	31.2	25.3	710

FD004	Random Forest	26.9	21.7	590

(Results vary depending on preprocessing & hyperparameters)

🚀 Future Improvements

Implement deep learning models (LSTMs/GRUs) for sequential dependencies

Add real-time streaming prediction capability

Integrate with cloud dashboards (Streamlit/Flask)

Hyperparameter optimization using Optuna or GridSearchCV

🤝 Contributing

Contributions are welcome!

Fork the repo

Create a feature branch (git checkout -b feature-name)

Commit changes (git commit -m "Added feature")

Push to branch (git push origin feature-name)

Open a Pull Request

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

👨‍💻 Author

Abhishek Kumar Gupta

📧 Email: 1289shek2004@gmail.com

🌐 GitHub: A289shek2004

💼 LinkedIn: https://www.linkedin.com/in/1289shek-gupta?

✨ This project was developed as part of the Upskill Campus Internship (Data Science domain).
