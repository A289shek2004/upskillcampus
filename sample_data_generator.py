import pandas as pd
import numpy as np

# Parameters
n_units = 5        # number of engines
n_cycles = 50      # cycles per engine
n_sensors = 10     # number of sensors

# Generate synthetic data
data = []
for unit in range(1, n_units + 1):
    for cycle in range(1, n_cycles + 1):
        sensors = np.random.normal(loc=100, scale=10, size=n_sensors)  # random sensor readings
        data.append([unit, cycle] + sensors.tolist())

# Create DataFrame
# Column names based on NASA C-MAPSS FD001 spec
col_names = [
    "unit", "cycle",
    "os1", "os2", "os3",  # operating settings
] + [f"s{i}" for i in range(1, 22)]  # 21 sensors

uploaded_file = st.file_uploader("📂 Upload engine sensor CSV/TXT", type=["csv", "txt"])

if uploaded_file:
    # Read space-separated file without headers
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=col_names)
    
    st.subheader("📄 Preview of Uploaded Data")
    st.write(df.head())

# Save CSV
df.to_csv("sample_engine_data.csv", index=False)

print("✅ Sample dataset generated: sample_engine_data.csv")
print(df.head())
