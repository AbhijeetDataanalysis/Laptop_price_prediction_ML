[README.md](https://github.com/user-attachments/files/34135095/README.md)
<div align="center">

# 💻 LaptopIQ — Laptop Price Predictor

### End-to-end Machine Learning project: Data Cleaning → Visualization → Model Comparison → Streamlit Web App

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

*Select a laptop's specifications and get an instant estimated price in ₹.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Results](#-model-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [How the Prediction Works](#-how-the-prediction-works)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Overview

**LaptopIQ** predicts the market price of a laptop from its hardware and software specifications (brand, type, RAM, SSD/HDD, CPU, GPU, screen sharpness, weight, OS, etc.).

The project covers the full data-science pipeline:

1. 🧹 **Clean** a raw 50,000-row laptop dataset
2. 📊 **Explore** which features drive price
3. 🤖 **Train & compare 3 algorithms** and pick the best one
4. 🌐 **Deploy** the model as an interactive Streamlit web app with a dashboard

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🏠 | **Overview tab** | 6 KPI cards + 5 charts (price distribution, price by brand, RAM, SSD, laptop-type mix) |
| 💰 | **Price Predictor tab** | Pick specs from dropdowns and get an instant price estimate |
| 🏆 | **Algorithm Lab tab** | Compares 3 algorithms with MAE, RMSE, R² and Accuracy, and highlights the best one |
| 🗂️ | **Dataset Explorer tab** | Browse the data and download it as CSV |
| 🔮 | **Log-price modelling** | Models are trained on `log(Price)` to handle the right-skewed price distribution |

---

## 🖼️ Demo

> click the link and see Demo 

| Price Predictor | Dashboard | Algorithm Lab |
|:---:|:---:|:---:|
| ![Predictor](assets/predictor.png) | ![Dashboard](assets/dashboard.png) | ![Algorithms](assets/algorithms.png) |

---

## 📦 Dataset

| Property | Value |
|---|---|
| Raw file | `laptop_dataset_50000_ml.csv` |
| Cleaned file | `cleaned_laptop_data.csv` |
| Rows (after cleaning) | **49,967** |
| Brands | 19 |
| Laptop types | 6 |
| Price range | ₹8,285 – ₹3,45,223 (mean ≈ ₹60,146) |

### Final features (12) and target

| Feature | Type | Description |
|---|---|---|
| `Company` | Categorical | Laptop brand |
| `TypeName` | Categorical | Notebook, Gaming, Ultrabook, 2-in-1, etc. |
| `Ram` | Numeric | RAM in GB |
| `Weight` | Numeric | Weight in kg |
| `Touchscreen` | Binary | 1 = Yes, 0 = No |
| `Ips` | Binary | 1 = IPS panel, 0 = No |
| `ppi` | Numeric | Pixels per inch (screen sharpness) |
| `Cpu brand` | Categorical | Intel Core i3 / i5 / i7, Other Intel, AMD |
| `HDD` | Numeric | HDD capacity in GB |
| `SSD` | Numeric | SSD capacity in GB |
| `Gpu brand` | Categorical | Intel, AMD, Nvidia |
| `os` | Categorical | Windows, Mac, Others/No OS/Linux |
| **`Price`** | **Target** | Laptop price in ₹ |

---

## 🔄 Project Workflow

```
Raw CSV ──► 01_Data_Cleaning ──► cleaned_laptop_data.csv
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              02_Data_Visualization             03_ML_Model
              (insights & charts)         (train 3 models, pick best)
                                                       │
                                          pipe.pkl + df.pkl
                                                       │
                                                       ▼
                                          LAPP.py (Streamlit app)
```

### 1️⃣ Data Cleaning — `01_Data_Cleaning.ipynb`
- Dropped the unused index column and filled missing values (**median** for numbers, **mode** for text)
- Converted `8GB` → `8` and `1.5kg` → `1.5`
- Extracted **Touchscreen**, **IPS** and resolution from the screen text, then computed **PPI**
- Simplified CPU names into 5 groups, GPU into brand, OS into 3 groups
- Split `Memory` into separate **HDD** and **SSD** columns (1 TB = 1000 GB)
- Removed the very rare ARM GPU rows

### 2️⃣ Data Visualization — `02_Data_Visualization.ipynb`
Key findings:
- 💾 **SSD, PPI and RAM** have the strongest relationship with price
- 🏢 **Razer, LG and MSI** are the most expensive brands; **Workstation** and **Gaming** are the most expensive types
- ✅ **IPS** and **Touchscreen** laptops cost more
- 📉 HDD size has little (even slightly negative) effect on price

### 3️⃣ Model Training — `03_ML_Model.ipynb`
- One-Hot Encoding for categorical columns inside a scikit-learn `Pipeline`
- 80 / 20 train–test split (`random_state=42`)
- Target: `log(Price)`, converted back with `exp()` so all metrics are in real ₹
- Best model is saved as `pipe.pkl`; cleaned data saved as `df.pkl`

---

## 🏆 Model Results

Three algorithms were trained and evaluated on the same unseen 20% test set (scores computed on the real price in ₹):

| Rank | Algorithm | R² Score | MAE (₹) | RMSE (₹) | Accuracy |
|:---:|---|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | **0.8869** | **7,750** | **12,617** | **86.99 %** |
| 🥈 | Gradient Boosting | 0.8589 | 9,216 | 14,096 | 84.41 % |
| 🥉 | Decision Tree | 0.8229 | 9,869 | 15,790 | 83.20 % |

> **Accuracy** = 100 − MAPE (mean absolute percentage error).

**✅ Winner: Random Forest** — it has the highest R² and the lowest error, because averaging many trees reduces the overfitting seen in a single Decision Tree.

The in-app **Algorithm Lab** tab re-evaluates three algorithms live (Linear Regression, Decision Tree and Random Forest) and ranks them by lowest RMSE. Exact numbers there can differ slightly from the table above because of different model settings.

> 💡 Results depend on the data split and hyperparameters, and may change if you retrain.

---

## 📁 Project Structure

```
📦 laptop-price-predictor
 ┣ 📜 01_Data_Cleaning.ipynb        # Clean the raw dataset
 ┣ 📜 02_Data_Visualization.ipynb   # EDA and charts
 ┣ 📜 03_ML_Model.ipynb             # Train, compare and save the model
 ┣ 📜 LAPP.py                       # Streamlit web application
 ┣ 📊 cleaned_laptop_data.csv       # Cleaned dataset
 ┣ 📦 df.pkl                        # Cleaned DataFrame (for dropdown options)
 ┣ 📦 pipe.pkl                      # Trained model pipeline (created by notebook 03)
 ┣ 📜 requirements.txt              # Python dependencies
 ┗ 📜 README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the model files *(skip if `pipe.pkl` and `df.pkl` already exist)*
Run the notebooks **in order**:

1. `01_Data_Cleaning.ipynb` → creates `cleaned_laptop_data.csv`
2. `02_Data_Visualization.ipynb` → *(optional)* charts and insights
3. `03_ML_Model.ipynb` → creates `pipe.pkl` and `df.pkl`

### 5. Run the app
```bash
streamlit run LAPP.py
```
The app opens at **http://localhost:8501** 🎉

> ⚠️ `pipe.pkl` and `df.pkl` must be in the **same folder** as `LAPP.py`.

---

## 🧠 How the Prediction Works

1. The user selects specs in the app.
2. **PPI** is calculated from the chosen resolution and screen size:

   `PPI = √(width² + height²) / screen size (inches)`

3. Yes/No options are converted to `1` / `0`.
4. The input is passed to the trained pipeline (One-Hot Encoding + Random Forest).
5. The model outputs `log(price)`, which is converted back with `exp()` to give the price in ₹.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | 🐍 Python |
| Data | 🐼 Pandas, 🔢 NumPy |
| Visualization | 📊 Matplotlib, Seaborn, Streamlit charts |
| Machine Learning | 🤖 scikit-learn (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) |
| Web App | 🎈 Streamlit |
| Notebooks | 📓 Jupyter |

---

## 🔭 Future Improvements

- [ ] Add XGBoost / LightGBM and hyperparameter tuning (GridSearchCV)
- [ ] Add feature-importance and SHAP explanations
- [ ] Add more features (screen size, battery, brand model series)
- [ ] Add a "compare two laptops" mode
- [ ] Deploy on Streamlit Community Cloud

---

## ⚠️ Disclaimer

The predicted price is an **estimate** generated by a machine learning model trained on a specific dataset. Real market prices vary by seller, region, time and offers.

---

## 👤 Abhijeet Pardhan

**Your Name**
- GitHub: https://github.com/AbhijeetDataanalysis
- LinkedIn:  www.linkedin.com/in/abhijeet-pardhan-34278b2a8

<div align="center">

⭐ If you found this project useful, please give it a star!

</div>
