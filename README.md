# 📈 ML Commodities Forecasting (VIP Sp26)

## Project Overview
This repository hosts the codebase for the Spring 2026 Vertically Integrated Project (VIP) focused on **Machine Learning in Commodities Markets**. 

The primary objective is to investigate and benchmark multiple machine learning architectures for time-series forecasting. We aim to predict price movements across various temporal horizons for key energy assets.

## 🎯 Objectives
* **Multi-Horizon Forecasting:** developing models to predict short-term (daily), medium-term (weekly), and long-term (monthly) price trends.
* **Model Benchmarking:** Comparing classical statistical methods against modern Deep Learning architectures.
* **Feature Engineering:** analyzing macroeconomic indicators, supply-chain constraints, and seasonality.

## 🛢️ Assets Under Investigation
We focus on the volatile energy sector, specifically:
* **Crude Oil** (WTI / Brent)
* **Natural Gas** (Henry Hub)
* **Electricity Markets**

## 🧠 Models & Architectures
*Current models under experimentation include:*
* **Baselines:** ARIMA, SARIMA
* **Tree-Based:** XGBoost, Random Forest, LightGBM
* **Deep Learning:** LSTM (Long Short-Term Memory), GRU, Transformer-based architectures (TimeGPT/PatchTST)

## 📂 Repository Structure
```text
ML_Commodities_VIP_Sp26/
├── data/                # Raw and processed datasets (ignored by git)
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── models/              # A slew of various models following an abstract class template
├── results/             # Plots and metrics from model runs
└── README.md
