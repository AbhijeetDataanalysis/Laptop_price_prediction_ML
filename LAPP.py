import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION & BRANDING
# =========================================================
st.set_page_config(
    page_title="LaptopIQ | Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1500px;}
    .hero {
        padding: 1.7rem 1.8rem; border-radius: 22px; color: white;
        background: radial-gradient(circle at 85% 15%, #0e7490 0, transparent 32%),
                    linear-gradient(120deg, #0b1220 0%, #172554 55%, #164e63 100%);
        margin-bottom: 1.1rem; border: 1px solid #31516c;
        box-shadow: 0 12px 30px rgba(0,0,0,.20);
    }
    .hero h1 {margin: 0; font-size: 2.5rem; letter-spacing: -.7px;}
    .hero p {margin: .55rem 0 0 0; color: #dbeafe; font-size: 1.03rem;}
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #151d2c, #101723);
        border: 1px solid #334155; padding: 15px 16px; border-radius: 16px;
        box-shadow: 0 5px 14px rgba(0,0,0,.12);
    }
    div[data-testid="stMetricLabel"] {color: #cbd5e1;}
    div[data-testid="stMetricValue"] {font-weight: 750;}
    div[data-testid="stTabs"] button {font-weight: 650; padding-top: .7rem; padding-bottom: .7rem;}
    div.stButton > button[kind="primary"] {
        border: 0; border-radius: 12px; min-height: 3rem;
        background: linear-gradient(90deg, #2563eb, #0891b2); font-weight: 700;
    }
    .small-muted {color: #9ca3af; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💻 LaptopIQ</h1>
  <p><b>LAPTOP PRICE INTELLIGENCE</b> &nbsp; · &nbsp; 📊 Market insights &nbsp; · &nbsp; 🧠 AI model benchmarking &nbsp; · &nbsp; 💰 Smart price estimates</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💻 LaptopIQ")
    st.caption("PRICE INTELLIGENCE STUDIO")
    st.divider()
    st.markdown("### 🧭 Navigation")
    st.write("Use the tabs at the top to explore insights, predict a price, compare algorithms, or inspect the dataset.")
    st.divider()
    st.markdown("### 🧪 Model selection")
    st.caption("The winning algorithm is chosen by the lowest RMSE on a held-out test set.")
    st.markdown("### 📌 Metrics guide")
    st.caption("MAE / RMSE: lower is better. R²: closer to 1 is generally better.")
    st.divider()
    st.caption("Built with Streamlit + scikit-learn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_PATH = os.path.join(BASE_DIR, "pipe.pkl")
DF_PATH = os.path.join(BASE_DIR, "df.pkl")


# =========================================================
# LOAD PROJECT FILES
# =========================================================
@st.cache_resource
def load_pickle(path, modified):
    with open(path, "rb") as file:
        return pickle.load(file)

if not os.path.exists(PIPE_PATH):
    st.error("Missing pipe.pkl. Place pipe.pkl in the same folder as this Python file.")
    st.stop()
if not os.path.exists(DF_PATH):
    st.error("Missing df.pkl. Place df.pkl in the same folder as this Python file.")
    st.stop()

try:
    pipe = load_pickle(PIPE_PATH, os.path.getmtime(PIPE_PATH))
    df = load_pickle(DF_PATH, os.path.getmtime(DF_PATH))
except Exception as exc:
    st.error(f"Could not load pipe.pkl or df.pkl: {exc}")
    st.stop()

if not isinstance(df, pd.DataFrame):
    st.error("df.pkl must contain a pandas DataFrame.")
    st.stop()

required_columns = [
    "Company", "TypeName", "Ram", "Weight", "Touchscreen", "Ips",
    "ppi", "Cpu brand", "HDD", "SSD", "Gpu brand", "os"
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    st.error("df.pkl is missing expected columns: " + ", ".join(missing))
    st.stop()

if "Price" not in df.columns:
    st.error("df.pkl must include a 'Price' column to calculate KPIs and compare algorithms.")
    st.stop()


# =========================================================
# MODEL COMPARISON: 3 REGRESSION ALGORITHMS
# =========================================================
@st.cache_resource
def evaluate_algorithms(dataframe, cache_key):
    data = dataframe.copy()
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["Price"])
    data = data[data["Price"] > 0]
    features = data[required_columns].copy()
    target = np.log(data["Price"].astype(float))

    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=18, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=20, random_state=42, n_jobs=-1
        )
    }

    rows = []
    for name, estimator in algorithms.items():
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        y_actual = np.exp(y_test.to_numpy())
        y_pred = np.maximum(np.exp(pred_log), 0)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        r2 = r2_score(y_actual, y_pred)
        nonzero = y_actual != 0
        mape = float(np.mean(np.abs((y_actual[nonzero] - y_pred[nonzero]) / y_actual[nonzero])) * 100) if nonzero.any() else np.nan
        rows.append({
            "Algorithm": name, "MAE": mae, "RMSE": rmse, "R²": r2,
            "Accuracy estimate (%)": max(0, 100 - mape) if pd.notna(mape) else np.nan
        })
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results

with st.spinner("🧠 Evaluating three algorithms on the laptop dataset..."):
    try:
        metrics = evaluate_algorithms(df, f'{len(df)}-{df["Price"].sum()}')
    except Exception as exc:
        metrics = None
        metrics_error = str(exc)

# =========================================================
# TABS
# =========================================================
tab_overview, tab_predict, tab_models, tab_data = st.tabs([
    "🏠 Overview", "💰 Price Predictor", "🏆 Algorithm Lab", "🗂️ Dataset Explorer"
])

# =========================================================
# TAB 1: OVERVIEW — 6 KPIs + 4 CHARTS
# =========================================================
with tab_overview:
    st.subheader("📌 Laptop Market Overview")
    st.caption("A quick summary of the laptop dataset and model performance.")

    prices = pd.to_numeric(df["Price"], errors="coerce").dropna()
    total_rows = len(df)
    unique_brands = df["Company"].nunique()
    avg_price = prices.mean() if len(prices) else 0
    median_price = prices.median() if len(prices) else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("💻 Laptop Listings", f"{total_rows:,}")
    k2.metric("🏢 Brands", f"{unique_brands:,}")
    k3.metric("💰 Average Price", f"₹{avg_price:,.0f}")
    k4, k5, k6 = st.columns(3)
    k4.metric("📍 Median Price", f"₹{median_price:,.0f}")
    if metrics is not None and not metrics.empty:
        best = metrics.iloc[0]
        k5.metric("🏆 Best Algorithm", best["Algorithm"])
        k6.metric("📉 Best RMSE", f'₹{best["RMSE"]:,.0f}')
    else:
        k5.metric("🏆 Best Algorithm", "Unavailable")
        k6.metric("📉 Best RMSE", "Unavailable")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 📊 1. Laptop Price Distribution")
        if not prices.empty:
            bins = pd.cut(prices, bins=min(20, max(5, prices.nunique())), duplicates="drop")
            histogram = bins.value_counts(sort=False).sort_index()
            histogram.index = histogram.index.astype(str)
            st.bar_chart(histogram)
            st.caption("Count of laptops in each price range.")
    with right:
        st.markdown("#### 🏢 2. Average Price by Brand")
        brand_prices = df.assign(Price=pd.to_numeric(df["Price"], errors="coerce"))
        brand_prices = brand_prices.groupby("Company")["Price"].mean().dropna().sort_values(ascending=False)
        st.bar_chart(brand_prices.head(12))
        st.caption("12 brands with the highest average listed price.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🧠 3. Average Price by RAM")
        ram_prices = df.assign(Price=pd.to_numeric(df["Price"], errors="coerce"))
        ram_prices = ram_prices.groupby("Ram")["Price"].mean().dropna().sort_index()
        st.line_chart(ram_prices)
        st.caption("How average price changes with RAM capacity.")
    with right:
        st.markdown("#### 💾 4. Average Price by SSD Capacity")
        ssd_prices = df.assign(Price=pd.to_numeric(df["Price"], errors="coerce"))
        ssd_prices = ssd_prices.groupby("SSD")["Price"].mean().dropna().sort_index()
        st.bar_chart(ssd_prices)
        st.caption("Average price for each SSD size.")

    st.markdown("#### 🧩 5. Laptop Mix by Type")
    type_counts = df["TypeName"].astype(str).value_counts().head(10)
    if not type_counts.empty:
        st.bar_chart(type_counts)
        st.caption("Number of listings for each laptop category.")


# =========================================================
# TAB 2: PRICE PREDICTOR
# =========================================================
with tab_predict:
    st.subheader("💰 Predict Your Laptop's Price")
    st.write("Select the laptop specifications and click **Predict Laptop Price**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💻 Basic Information")
        company = st.selectbox("🏢 Company / Brand", sorted(df["Company"].dropna().unique().tolist()))
        laptop_type = st.selectbox("🖥️ Laptop Type", sorted(df["TypeName"].dropna().unique().tolist()))
        ram_options = sorted(pd.to_numeric(df["Ram"], errors="coerce").dropna().unique().tolist())
        ram_default = ram_options.index(8) if 8 in ram_options else 0
        ram = st.selectbox("🧠 RAM (GB)", ram_options, index=ram_default)
        weight = st.number_input("⚖️ Weight (kg)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
        touchscreen = st.selectbox("👆 Touchscreen", ["No", "Yes"])
        ips = st.selectbox("🖥️ IPS Display", ["No", "Yes"])
        screen_size = st.number_input("📐 Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)

    with col2:
        st.markdown("### ⚙️ Hardware & Software")
        resolutions = [
            "1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
            "2880x1800", "2560x1600", "2560x1440", "2304x1440"
        ]
        resolution = st.selectbox("🖼️ Screen Resolution", resolutions, index=0)
        cpu = st.selectbox("⚡ CPU Brand", sorted(df["Cpu brand"].dropna().unique().tolist()))
        hdd_options = sorted(pd.to_numeric(df["HDD"], errors="coerce").dropna().unique().tolist())
        hdd = st.selectbox("💽 HDD (GB)", hdd_options)
        ssd_options = sorted(pd.to_numeric(df["SSD"], errors="coerce").dropna().unique().tolist())
        ssd_default = ssd_options.index(256) if 256 in ssd_options else 0
        ssd = st.selectbox("🚀 SSD (GB)", ssd_options, index=ssd_default)
        gpu = st.selectbox("🎮 GPU Brand", sorted(df["Gpu brand"].dropna().unique().tolist()))
        operating_system = st.selectbox("🪟 Operating System", sorted(df["os"].dropna().unique().tolist()))

    st.markdown("### 📋 Selected Configuration")
    specs = st.columns(4)
    specs[0].metric("Brand", str(company))
    specs[1].metric("RAM", f"{ram:g} GB")
    specs[2].metric("SSD", f"{ssd:g} GB")
    specs[3].metric("GPU", str(gpu))
    specs2 = st.columns(4)
    specs2[0].metric("CPU", str(cpu))
    specs2[1].metric("HDD", f"{hdd:g} GB")
    specs2[2].metric("Screen", f'{screen_size:.1f}"')
    specs2[3].metric("Weight", f"{weight:.1f} kg")

    if st.button("🔮 Predict Laptop Price", type="primary", use_container_width=True):
        touch_value = 1 if touchscreen == "Yes" else 0
        ips_value = 1 if ips == "Yes" else 0
        x_res, y_res = map(int, resolution.split("x"))
        ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
        query = pd.DataFrame([{
            "Company": company, "TypeName": laptop_type, "Ram": ram, "Weight": weight,
            "Touchscreen": touch_value, "Ips": ips_value, "ppi": ppi,
            "Cpu brand": cpu, "HDD": hdd, "SSD": ssd, "Gpu brand": gpu, "os": operating_system
        }])
        try:
            prediction = int(round(np.exp(pipe.predict(query)[0])))
            st.success("✅ Prediction generated successfully!")
            st.markdown(
                f"<div style='text-align:center;padding:1.2rem;border-radius:16px;background:linear-gradient(115deg,#102033,#075985);'>"
                f"<div style='font-size:1.1rem;color:#dbeafe;'>💰 Estimated Laptop Price</div>"
                f"<div style='font-size:2.8rem;font-weight:750;color:white;'>₹ {prediction:,}</div></div>",
                unsafe_allow_html=True
            )
            st.caption("This is an estimate based on the trained pipeline and selected specifications.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# TAB 3: ALGORITHM LAB — COMPARE 3 ALGORITHMS
# =========================================================
with tab_models:
    st.subheader("🏆 Machine Learning Algorithm Lab")
    st.write("Three regression algorithms are trained on the same 80/20 train/test split using the available dataset features. The best model is selected by the lowest RMSE.")
    if metrics is None:
        st.error(f"Could not calculate model comparison: {metrics_error}")
    else:
        best = metrics.iloc[0]
        st.success(f'🏆 Best Algorithm: **{best["Algorithm"]}** — lowest RMSE of ₹{best["RMSE"]:,.0f}')
        st.dataframe(
            metrics.style.format({
                "MAE": "₹{:,.0f}", "RMSE": "₹{:,.0f}",
                "R²": "{:.4f}", "Accuracy estimate (%)": "{:.2f}%"
            }),
            use_container_width=True, hide_index=True
        )
        st.info(
            f"💡 **How to read this result:** {best['Algorithm']} had the lowest RMSE on this test split. "
            "This does not guarantee it will be best on every dataset or future market prices."
        )
        st.markdown("#### 📉 5. RMSE by Algorithm (lower is better)")
        st.bar_chart(metrics.set_index("Algorithm")[["RMSE"]])
        st.markdown("#### 📊 6. MAE by Algorithm (lower is better)")
        st.bar_chart(metrics.set_index("Algorithm")[["MAE"]])
        left, right = st.columns(2)
        with left:
            st.markdown("#### 📈 R² Score by Algorithm (higher is better)")
            st.bar_chart(metrics.set_index("Algorithm")[["R²"]])
        with right:
            st.markdown("#### 🎯 Accuracy Estimate by Algorithm")
            st.bar_chart(metrics.set_index("Algorithm")[["Accuracy estimate (%)"]])
        st.caption("Accuracy estimate = 100 − MAPE. It is a regression metric, not classification accuracy. Results depend on the dataset and test split.")


# =========================================================
# TAB 4: DATASET EXPLORER
# =========================================================
with tab_data:
    st.subheader("🗂️ Laptop Dataset Explorer")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    st.markdown("#### 🔎 Filter and preview laptop records")
    brand_options = sorted(df["Company"].dropna().astype(str).unique().tolist())
    chosen_brands = st.multiselect("Filter by brand", brand_options)
    view = df[df["Company"].astype(str).isin(chosen_brands)] if chosen_brands else df
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download dataset preview",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="laptop_dataset.csv",
        mime="text/csv"
    )

st.divider()
st.caption("💻 LaptopIQ | Laptop Price Intelligence • Built with Streamlit, pandas and scikit-learn")
