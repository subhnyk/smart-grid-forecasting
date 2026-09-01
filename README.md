# ⚡ Adaptive Smart Grid Load Forecasting & Autonomous Battery Dispatch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792EE5.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Project Overview

This project is an **end-to-end AI-based smart grid system** designed to solve two important energy management problems:

1. **Predict future electricity demand**
2. **Automatically decide when a battery should charge or discharge**

In simple words, the system first tries to answer:

> **"How much electricity will be needed in the future?"**

After predicting the electricity demand, it then answers:

> **"What is the best action for the battery right now?"**

For example:

- If electricity demand is expected to increase, the battery may discharge energy.
- If electricity prices are low or excess renewable energy is available, the battery may charge.
- The system also considers battery health to avoid unnecessary damage.

The project combines **Deep Learning**, **Reinforcement Learning**, **API deployment**, and an **interactive dashboard** into one complete system.

---

# 🎯 Project Objectives

The main objectives of this project are:

### 1. Forecast Electricity Demand

Use historical electricity and weather data to predict future **net electricity load**.

The forecasting model provides predictions for multiple future time periods.

### 2. Handle Prediction Uncertainty

Instead of providing only one prediction, the system generates three prediction levels:

- **P10** – Lower demand estimate
- **P50** – Most likely or median estimate
- **P90** – Higher demand estimate

This helps grid operators understand the possible range of future electricity demand.

### 3. Optimize Battery Operations

Use Reinforcement Learning to automatically determine whether the battery should:

- 🔋 Charge
- ⏸️ Remain idle
- ⚡ Discharge

The goal is to improve energy efficiency and economic performance.

### 4. Protect Battery Health

The reward system penalizes harmful battery behavior such as:

- Overcharging
- Excessive deep discharge
- Unnecessary charge/discharge cycles
- Thermal stress

This encourages the AI agent to make decisions that are beneficial in both the short term and long term.

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **Temporal Fusion Transformer (TFT)** | Electricity load forecasting |
| **PyTorch / PyTorch Lightning** | Deep learning model development and training |
| **Proximal Policy Optimization (PPO)** | Battery decision-making using Reinforcement Learning |
| **Stable-Baselines3** | PPO model implementation |
| **Gymnasium** | Custom battery simulation environment |
| **FastAPI** | Creating REST API endpoints |
| **Streamlit** | Building the interactive dashboard |
| **Plotly** | Data visualization |
| **EIA / Open-Meteo** | Electricity and weather data sources |

---

# 🏗️ How the Complete System Works

The project follows the pipeline below:

```text
                ┌──────────────────────┐
                │     Data Sources     │
                │                      │
                │ Electricity + Weather│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Data Ingestion &    │
                │  Feature Engineering │
                └──────────┬───────────┘
                           │
                           ▼
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
┌──────────────────────┐        ┌──────────────────────┐
│  TFT Forecast Model  │        │ PPO Battery Agent    │
│                      │        │                      │
│ Predicts future      │        │ Decides whether to   │
│ electricity demand   │        │ charge or discharge  │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    FastAPI Service   │
                │                      │
                │ Provides predictions │
                │ and battery actions  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Streamlit Dashboard  │
                │                      │
                │ Visualizes forecasts │
                │ and battery status   │
                └──────────────────────┘
```

---

# 🔄 Step-by-Step Workflow

## Step 1: Collect Data

The system collects information from different sources.

The data may include:

- Historical electricity demand
- Electricity prices
- Weather conditions
- Temperature
- Time and date information
- Renewable energy availability

This raw data is stored inside the:

```text
data/raw/
```

directory.

---

## Step 2: Process and Prepare the Data

Raw data cannot always be directly used by a machine learning model.

Therefore, the project performs feature engineering and preprocessing.

Examples include:

- Handling missing values
- Normalizing numerical features
- Creating time-based features
- Creating cyclic features for hours, days, and months
- Preparing the target variable for forecasting

For example, time features can help the model understand that:

- 23:00 and 00:00 are close to each other
- Sunday and Monday have a repeating weekly relationship
- Electricity demand may change based on season

The processed data is stored inside:

```text
data/processed/
```

---

# 📈 Step 3: Forecast Future Electricity Demand

The project uses a **Temporal Fusion Transformer (TFT)**.

TFT is a deep learning model designed for time-series forecasting.

The model learns patterns from historical data and predicts future electricity demand.

Instead of generating only one prediction, it produces multiple quantiles:

| Prediction | Meaning |
|---|---|
| **P10** | Lower possible demand |
| **P50** | Most likely demand |
| **P90** | Higher possible demand |

For example:

```text
P10 = 850 MW
P50 = 1000 MW
P90 = 1150 MW
```

This means the expected electricity demand is approximately **1000 MW**, but the actual demand may vary between lower and higher values.

This uncertainty information is useful when making operational decisions.

The trained model is stored as:

```text
models/tft_model.ckpt
```

---

# 🔋 Step 4: Train the Battery Decision Agent

The project uses **Proximal Policy Optimization (PPO)**, a Reinforcement Learning algorithm.

The PPO agent learns by interacting with a simulated battery environment.

At each step, the agent observes information such as:

- Current electricity demand
- Forecasted demand
- Electricity price
- Battery state of charge
- Available renewable energy
- Previous system conditions

Based on this information, the agent chooses an action.

Possible actions include:

```text
Charge
   ↓
Idle
   ↓
Discharge
```

The agent receives a **reward** based on the quality of its decision.

For example:

### Good Decision

```text
Low electricity price
        ↓
Battery charges
        ↓
Electricity demand later increases
        ↓
Battery discharges
        ↓
Positive reward
```

### Bad Decision

```text
Battery repeatedly charges and discharges
        ↓
Battery degradation increases
        ↓
Negative reward
```

Through many training episodes, the PPO agent learns a strategy for making better battery decisions.

The trained PPO model is stored as:

```text
models/ppo_battery_dispatch.zip
```

---

# 🛡️ Battery Health and Degradation Awareness

Battery performance is not only about making profit or saving energy.

Frequent charging and discharging can reduce the lifespan of a battery.

Therefore, the reinforcement learning reward function considers battery health.

The agent can be penalized for:

- Excessive cycling
- Very low battery charge
- Overcharging
- Deep discharge
- Thermal stress

This creates a balance between:

```text
Economic Performance
        +
Energy Optimization
        +
Battery Health
```

The goal is to make the battery operate intelligently without unnecessarily reducing its lifespan.

---

# ✨ Key Features

## 📊 Multi-Horizon Load Forecasting

The TFT model predicts electricity demand for future time periods instead of only the current moment.

It provides:

- P10 forecast
- P50 forecast
- P90 forecast

This helps represent uncertainty in future electricity demand.

---

## 🤖 Autonomous Battery Control

The PPO reinforcement learning agent automatically learns when to:

- Charge the battery
- Keep the battery idle
- Discharge the battery

The agent improves its strategy through interaction with the simulation environment.

---

## 💰 Energy Arbitrage

The system attempts to take advantage of changes in electricity prices.

A simplified example:

```text
Low Price
    ↓
Charge Battery
    ↓
Store Energy
    ↓
High Price
    ↓
Discharge Battery
```

This process is known as **energy arbitrage**.

---

## 🛡️ Battery Health Protection

The system considers battery degradation while making decisions.

This helps prevent the agent from maximizing short-term benefits at the cost of damaging the battery.

---

## 🌐 Production REST API

A FastAPI service allows other applications to communicate with the AI system.

The API can be used to request:

- Electricity demand forecasts
- Battery status
- Battery actions
- System health information

---

## 📊 Interactive Dashboard

The Streamlit dashboard provides a user-friendly interface for monitoring the system.

Possible information displayed includes:

- Electricity demand forecasts
- Battery state of charge
- Battery actions
- Model predictions
- Performance metrics
- Operational results

---

# 📂 Project Structure

```text
smart-grid-forecasting/
│
├── data/
│   │
│   ├── raw/
│   │   └── Raw electricity and weather data
│   │
│   └── processed/
│       └── Cleaned and feature-engineered data
│
├── models/
│   │
│   ├── tft_model.ckpt
│   │   └── Trained Temporal Fusion Transformer model
│   │
│   └── ppo_battery_dispatch.zip
│       └── Trained PPO battery decision model
│
├── src/
│   │
│   ├── ingestion/
│   │   └── Scripts for collecting raw data
│   │
│   ├── features/
│   │   └── Data preprocessing and feature engineering
│   │
│   ├── models/
│   │   └── Model definitions and training scripts
│   │
│   ├── rl_env/
│   │   └── Custom Gymnasium environment for battery simulation
│   │
│   └── serving/
│       │
│       ├── api.py
│       │   └── FastAPI application
│       │
│       └── dashboard.py
│           └── Streamlit dashboard
│
├── tests/
│   └── Tests for data, models, and API endpoints
│
├── .env
│   └── Environment variables
│
├── requirements.txt
│   └── Required Python libraries
│
└── README.md
    └── Project documentation
```

---

# 🚀 Getting Started

## Step 1: Install Python

Make sure you have **Python 3.10 or later** installed.

You can check your Python version using:

```powershell
python --version
```

---

## Step 2: Clone the Repository

Open PowerShell or a terminal and run:

```powershell
git clone https://github.com/your-username/smart-grid-forecasting.git
```

Move into the project folder:

```powershell
cd smart-grid-forecasting
```

> Replace `your-username` with your actual GitHub username.

---

## Step 3: Create a Virtual Environment

A virtual environment keeps the project's Python libraries separate from other projects.

Create the environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

---

## Step 4: Install Required Libraries

Once the virtual environment is activated, install all dependencies:

```powershell
pip install -r requirements.txt
```

---

# ▶️ Running the Machine Learning Pipeline

The project pipeline should generally be executed in the following order.

## 1. Collect Raw Data

```powershell
python src/ingestion/run_ingestion.py
```

This step downloads or collects the required electricity and weather data.

---

## 2. Build Features

```powershell
python src/features/build_features.py
```

This step prepares the raw data for machine learning.

Examples include:

- Cleaning data
- Creating time-based features
- Normalizing values
- Preparing model inputs

---

## 3. Train the TFT Forecasting Model

```powershell
python src/models/train_tft.py
```

This trains the Temporal Fusion Transformer to predict future electricity demand.

---

## 4. Train the PPO Battery Agent

```powershell
python src/models/train_rl_dispatch.py
```

This trains the reinforcement learning agent to make battery charge and discharge decisions.

---

# 🌐 Running the Application

The application consists of two main components:

1. **FastAPI Backend**
2. **Streamlit Dashboard**

They should be started in separate terminals.

---

# Terminal 1: Start the FastAPI Backend

Run:

```powershell
python -m uvicorn src.serving.api:app --reload --port 8000
```

The API server will run locally.

After starting the server, you can access:

### API Documentation

```text
http://localhost:8000/docs
```

FastAPI automatically provides an interactive interface where you can test API endpoints.

### Health Check

```text
http://localhost:8000/health
```

This endpoint can be used to check whether the API service is running correctly.

---

# Terminal 2: Start the Streamlit Dashboard

Open another terminal, activate the virtual environment again, and run:

```powershell
streamlit run src/serving/dashboard.py
```

The dashboard will usually be available at:

```text
http://localhost:8501
```

Open this address in your browser to access the application interface.

---

# 📊 Model Evaluation

The project evaluates different parts of the system separately.

| Component | What is Measured | Target / Output |
|---|---|---|
| **TFT Forecasting Model** | Accuracy of electricity demand predictions | WAPE below 4.2% |
| **PPO Battery Agent** | Improvement in battery arbitrage strategy | +18.4% compared with a rule-based baseline |
| **FastAPI Service** | Model inference response time | Below 45 ms per request |

## What is WAPE?

**WAPE** stands for **Weighted Absolute Percentage Error**.

It is used to measure forecasting accuracy.

A lower WAPE value generally indicates better forecasting performance.

For example:

```text
Lower WAPE
     ↓
Prediction is closer to actual electricity demand
```

---

# 🔮 Example of the Complete System

Suppose the system receives the following information:

```text
Current Time: 2:00 PM
Current Demand: Moderate
Electricity Price: Low
Weather: Sunny
Battery Charge: 40%
```

The TFT forecasting model predicts:

```text
P10: 900 MW
P50: 1100 MW
P90: 1250 MW
```

This suggests that electricity demand may increase.

The PPO agent analyzes:

- Current battery level
- Future demand forecast
- Electricity price
- Expected system conditions

The agent may decide:

```text
ACTION → CHARGE BATTERY
```

Later, when electricity demand and prices increase:

```text
ACTION → DISCHARGE BATTERY
```

The goal is to improve energy management while also protecting the battery from excessive degradation.

---

# 🧪 Testing

The `tests/` directory contains tests for different parts of the application.

Testing can help verify that:

- Data follows the expected format
- Features are created correctly
- API endpoints work properly
- Models can be loaded successfully

Example:

```powershell
pytest
```

---

# 🎓 Who Is This Project For?

This project can be useful for:

- Data Science students
- Machine Learning engineers
- Deep Learning learners
- Reinforcement Learning learners
- Energy analytics projects
- Smart grid research
- Battery optimization research
- AI portfolio projects

It is also a good example of how multiple technologies can be combined into a complete machine learning application.

---

# 🛣️ Future Improvements

Possible future improvements include:

- Adding real-time streaming data
- Adding electricity price forecasting
- Supporting multiple battery systems
- Adding renewable energy forecasting
- Deploying the API to a cloud platform
- Adding Docker support
- Adding CI/CD pipelines
- Adding model monitoring
- Adding automated model retraining
- Improving dashboard visualizations
- Adding user authentication

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute the project according to the terms of the MIT License.

See the `LICENSE` file for more details.

---

# 👨‍💻 Author

**Your Name**

If you found this project useful, consider giving the repository a ⭐ on GitHub!

---

## ⭐ Final Summary

This project combines several modern AI technologies into one complete smart energy management system:

```text
Electricity + Weather Data
            ↓
     Data Processing
            ↓
   TFT Load Forecasting
            ↓
  Future Demand Prediction
            ↓
 PPO Battery Decision Agent
            ↓
 Charge / Idle / Discharge
            ↓
       FastAPI Backend
            ↓
    Streamlit Dashboard
```

The main purpose of the system is to **predict future electricity demand and intelligently control a battery system based on expected grid conditions, economic opportunities, and battery health considerations**.