
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import secrets

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# PAGE SETUP AND EXISTING CSS DESIGN
# =========================================================
st.set_page_config(
    page_title="CarPredict | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.html("""
<style>
.stApp { background:#f4f8fc; color:#183b63; }
.main .block-container { max-width:1320px; padding-top:1.4rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #dce6f2; }
[data-testid="stSidebar"] * { color:#183b63; }
.brand { text-align:center; padding:8px 4px 22px; }
.brand-icon { font-size:48px; }
.brand-name { color:#1261a0; font-size:25px; font-weight:850; }
.brand-sub { color:#718096; font-size:13px; }
.hero { padding:34px 38px; border-radius:22px; color:white;
        background:linear-gradient(120deg,#1473e6,#08a6b5);
        box-shadow:0 12px 28px rgba(20,115,230,.17); margin-bottom:22px; }
.hero h1 { color:white; font-size:clamp(26px,3vw,39px); margin:0 0 9px; }
.hero p { color:white; font-size:16px; margin:0; opacity:.96; }
.panel { background:white; border:1px solid #e1eaf3; border-radius:17px;
         padding:23px; box-shadow:0 5px 16px rgba(30,64,100,.05); margin-bottom:18px; }
.panel-title { color:#1261a0; font-size:22px; font-weight:800; margin-bottom:8px; }
.muted { color:#64748b; line-height:1.7; }
.metric { background:#fff; border:1px solid #e1eaf3; border-radius:15px;
          padding:18px 12px; text-align:center; height:100%;
          box-shadow:0 4px 13px rgba(30,64,100,.04); }
.metric-icon { font-size:25px; }
.metric-label { color:#64748b; font-size:13px; font-weight:650; margin-top:5px; }
.metric-value { color:#1261a0; font-size:25px; font-weight:850; margin-top:3px; }
.feature { background:#fff; border:1px solid #e1eaf3; border-radius:15px;
           padding:19px; min-height:145px; height:100%; }
.feature-icon { font-size:28px; }
.feature-title { color:#1261a0; font-weight:800; font-size:17px; margin:7px 0 5px; }
.feature-copy { color:#64748b; font-size:14px; line-height:1.5; }
.price-result { background:linear-gradient(120deg,#e0f2fe,#ecfeff);
                border:2px solid #38bdf8; border-radius:18px; padding:25px;
                text-align:center; margin:15px 0; }
.price-caption { color:#0369a1; font-weight:750; font-size:16px; }
.price-value { color:#075985; font-weight:900; font-size:40px; margin:5px 0; }
.status { background:#dcfce7; color:#166534; border:1px solid #bbf7d0;
          border-radius:10px; padding:10px 12px; font-weight:750; text-align:center; }
.small-note { color:#64748b; font-size:12px; line-height:1.6; }
div.stButton > button { border-radius:10px; border:0; font-weight:700; min-height:2.65rem; }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#1473e6,#08a6b5); color:white; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e1eaf3; padding:12px 15px; border-radius:13px; }
label, .stSelectbox label, .stNumberInput label, .stSlider label { color:#183b63 !important; font-weight:650 !important; }
[data-testid="stDataFrame"] { border:1px solid #e1eaf3; border-radius:12px; overflow:hidden; }
.footer { text-align:center; color:#94a3b8; font-size:12px; padding:24px 0 5px; }
</style>
""")

# =========================================================
# DATASET AND MODEL
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("used_car_price_dataset.csv")


@st.cache_resource
def train_model(data):
    encoded = pd.get_dummies(
        data,
        columns=["Fuel_Type", "Seller_Type", "Transmission"],
        drop_first=True
    )

    encoded = encoded.drop(columns=["Car_Name"], errors="ignore")
    X = encoded.drop(columns=["Selling_Price"])
    y = encoded["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return model, X, X_train, X_test, y_test, predictions, mae, rmse, r2


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Keep used_car_price_dataset.csv "
        "in the same folder as app.py."
    )
    st.stop()

required_columns = {
    "Year", "Selling_Price", "Present_Price", "Kms_Driven",
    "Fuel_Type", "Seller_Type", "Transmission", "Owner"
}

missing = required_columns - set(df.columns)
if missing:
    st.error(f"Missing dataset columns: {', '.join(sorted(missing))}")
    st.stop()

try:
    model, X, X_train, X_test, y_test, predictions, mae, rmse, r2 = train_model(df)
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

score = r2 * 100

actual = y_test.to_numpy()
valid = actual != 0
if valid.any():
    within_10 = np.mean(
        np.abs((actual[valid] - predictions[valid]) / actual[valid]) <= 0.10
    ) * 100
else:
    within_10 = 0.0

# =========================================================
# ACCOUNT DATABASE - CORRECT INDENTATION
# =========================================================
DB_NAME = "carpredict_users.db"


def get_db_connection():
    return sqlite3.connect(DB_NAME, timeout=10)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200000
    ).hex()

    return salt, password_hash


def initialize_user_database():
    # Everything inside this with block is indented correctly.
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        existing = connection.execute(
            "SELECT 1 FROM users WHERE username = ?",
            ("admin",)
        ).fetchone()

        if existing is None:
            salt, password_hash_value = hash_password("1234")
            connection.execute(
                """
                INSERT INTO users (username, salt, password_hash)
                VALUES (?, ?, ?)
                """,
                ("admin", salt, password_hash_value)
            )


def create_user(username, password):
    salt, password_hash_value = hash_password(password)

    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (username, salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (username.strip(), salt, password_hash_value)
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with get_db_connection() as connection:
        user = connection.execute(
            """
            SELECT salt, password_hash
            FROM users
            WHERE username = ?
            """,
            (username.strip(),)
        ).fetchone()

    if user is None:
        return False

    salt, saved_hash = user
    _, entered_hash = hash_password(password, salt)
    return secrets.compare_digest(entered_hash, saved_hash)


initialize_user_database()

# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_username" not in st.session_state:
    st.session_state.current_username = ""

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# =========================================================
# LOGIN AND CREATE ACCOUNT
# =========================================================
if not st.session_state.logged_in:
    st.markdown("""
    <div style="max-width:620px;margin:55px auto 20px;background:#fff;
    padding:34px;border-radius:22px;border:1px solid #e1eaf3;
    box-shadow:0 12px 28px rgba(30,64,100,.08);text-align:center;">
      <div style="font-size:55px">🚗</div>
      <div style="font-size:32px;font-weight:850;color:#1261a0">CarPredict</div>
      <div style="color:#64748b;margin-top:5px">
        Used Car Price Prediction &amp; Analysis
      </div>
      <div style="color:#64748b;margin-top:18px;line-height:1.7">
        Explore vehicle data, understand model performance, and estimate a used car's price.
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        mode = st.radio(
            "Account options",
            ["Login", "Create New Account"],
            horizontal=True
        )

        if mode == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_clicked = st.form_submit_button(
                    "🔐 Login", type="primary", width="stretch"
                )

            if login_clicked:
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.current_username = username.strip()
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

            st.caption("Demo login: admin / 1234")

        else:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username")
                new_password = st.text_input("Create password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                signup_clicked = st.form_submit_button(
                    "✨ Create Account", type="primary", width="stretch"
                )

            if signup_clicked:
                username_clean = new_username.strip()

                if not 3 <= len(username_clean) <= 30:
                    st.error("Username must be between 3 and 30 characters.")
                elif not all(c.isalnum() or c in "._-" for c in username_clean):
                    st.error("Username may contain letters, numbers, dots, underscores, and hyphens only.")
                elif len(new_password) < 8:
                    st.error("Password must contain at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif create_user(username_clean, new_password):
                    st.success("Account created! Select Login to sign in.")
                else:
                    st.error("That username already exists.")

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
pages = [
    "🏠 Home",
    "🔮 Price Prediction",
    "📊 Data Analysis",
    "📈 Model Performance",
    "ℹ️ About Project"
]

with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">🚗</div>
      <div class="brand-name">CarPredict</div>
      <div class="brand-sub">Used Car Price Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"**Signed in as:** {st.session_state.current_username}")
    st.markdown("**NAVIGATION**")

    selected_page = st.radio(
        "Choose a page",
        pages,
        index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
        label_visibility="collapsed"
    )
    st.session_state.page = selected_page

    st.divider()
    st.markdown('<div class="status">✓ Model Ready</div>', unsafe_allow_html=True)
    st.metric("R²-based score", f"{score:.2f}%")

    if st.button("↻ Refresh app", width="stretch"):
        st.rerun()

    if st.button("🚪 Logout", width="stretch"):
        st.session_state.logged_in = False
        st.session_state.current_username = ""
        st.rerun()

# =========================================================
# REUSABLE UI FUNCTIONS
# =========================================================
def page_hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def metric_card(icon, label, value):
    st.markdown(
        f"""
        <div class="metric">
          <div class="metric-icon">{icon}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


page = st.session_state.page

# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    page_hero(
        "🚗 Used Car Price Prediction",
        "Estimate used-car prices using machine learning."
    )

    st.markdown("""
    <div class="panel">
      <div class="panel-title">📌 Project Overview</div>
      <div class="muted">
        The Used Car Price Prediction and Analysis System uses machine
        learning to estimate a car's selling price. It considers year,
        present price, kilometres driven, fuel type, seller type,
        transmission, and previous owners.
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("🚘", "Dataset Records", f"{len(df)}"),
        ("🎯", "R²-based Score", f"{score:.2f}%"),
        ("📏", "Mean Absolute Error", f"{mae:.3f}"),
        ("📊", "R² Score", f"{r2:.3f}")
    ]

    for col, card in zip(cols, cards):
        with col:
            metric_card(*card)

    st.markdown("### ✨ Explore CarPredict")
    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="feature"><div class="feature-icon">🔮</div>
        <div class="feature-title">Predict a price</div>
        <div class="feature-copy">Enter car details to estimate a selling price.</div></div>
        """, unsafe_allow_html=True)
        if st.button("Open Price Prediction", width="stretch"):
            st.session_state.page = "🔮 Price Prediction"
            st.rerun()

    with b:
        st.markdown("""
        <div class="feature"><div class="feature-icon">📊</div>
        <div class="feature-title">Explore the dataset</div>
        <div class="feature-copy">Filter records and explore interactive charts.</div></div>
        """, unsafe_allow_html=True)
        if st.button("Open Data Analysis", width="stretch"):
            st.session_state.page = "📊 Data Analysis"
            st.rerun()

    with c:
        st.markdown("""
        <div class="feature"><div class="feature-icon">📈</div>
        <div class="feature-title">Model performance</div>
        <div class="feature-copy">Explore model metrics and feature importance.</div></div>
        """, unsafe_allow_html=True)
        if st.button("Open Model Performance", width="stretch"):
            st.session_state.page = "📈 Model Performance"
            st.rerun()

    st.markdown("""
    <div class="panel"><div class="panel-title">⚙️ How it works</div>
    <div class="muted">
    1. Load vehicle data → 2. Prepare features → 3. Train Random Forest →
    4. Evaluate the model → 5. Predict selling price.
    </div></div>
    """, unsafe_allow_html=True)

# =========================================================
# PRICE PREDICTION
# =========================================================
elif page == "🔮 Price Prediction":
    page_hero(
        "🔮 Car Price Prediction",
        "Enter vehicle details to estimate its selling price."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        year_min = max(1990, int(df["Year"].min()))
        year_max = max(2026, int(df["Year"].max()))

        with left:
            year = st.slider(
                "Manufacturing year", year_min, year_max,
                min(2018, year_max)
            )

            present_min = max(0.1, float(df["Present_Price"].min()))
            present_max = max(present_min + 0.1, float(df["Present_Price"].max()))

            present_price = st.number_input(
                "Present price (Lakh)",
                min_value=present_min,
                max_value=present_max,
                value=float(min(8.5, present_max)),
                step=0.1
            )

            kms = st.number_input(
                "Kilometres driven",
                min_value=0,
                max_value=max(500000, int(df["Kms_Driven"].max())),
                value=30000,
                step=1000
            )

            owner = st.selectbox("Previous owners", [0, 1, 2, 3])

        with right:
            fuels = sorted(df["Fuel_Type"].dropna().astype(str).unique())
            sellers = sorted(df["Seller_Type"].dropna().astype(str).unique())
            transmissions = sorted(df["Transmission"].dropna().astype(str).unique())

            fuel = st.selectbox(
                "Fuel type", fuels,
                index=fuels.index("Petrol") if "Petrol" in fuels else 0
            )

            seller = st.selectbox(
                "Seller type", sellers,
                index=sellers.index("Dealer") if "Dealer" in sellers else 0
            )

            transmission = st.selectbox(
                "Transmission", transmissions,
                index=transmissions.index("Manual") if "Manual" in transmissions else 0
            )

        predict_clicked = st.form_submit_button(
            "💰 Predict Selling Price",
            type="primary",
            width="stretch"
        )

    if predict_clicked:
        input_row = {
            "Year": year,
            "Present_Price": present_price,
            "Kms_Driven": kms,
            "Owner": owner
        }

        for column in X.columns:
            if column.startswith("Fuel_Type_"):
                input_row[column] = int(fuel == column.replace("Fuel_Type_", ""))
            elif column.startswith("Seller_Type_"):
                input_row[column] = int(seller == column.replace("Seller_Type_", ""))
            elif column.startswith("Transmission_"):
                input_row[column] = int(transmission == column.replace("Transmission_", ""))

        input_data = pd.DataFrame([input_row]).reindex(
            columns=X.columns, fill_value=0
        ).astype(float)

        price = float(model.predict(input_data)[0])

        result = {
            "price": price,
            "year": year,
            "present_price": present_price,
            "kms": kms,
            "owner": owner,
            "fuel": fuel,
            "seller": seller,
            "transmission": transmission
        }

        st.session_state.last_prediction = result
        st.session_state.prediction_history.insert(0, {
            "Predicted Price (Lakh)": round(price, 2),
            "Year": year,
            "Present Price (Lakh)": present_price,
            "Kms Driven": kms,
            "Fuel": fuel,
            "Seller": seller,
            "Transmission": transmission
        })
        st.session_state.prediction_history = st.session_state.prediction_history[:10]

    result = st.session_state.last_prediction

    if result:
        st.markdown(f"""
        <div class="price-result">
          <div class="price-caption">Estimated Selling Price</div>
          <div class="price-value">₹ {result['price']:.2f} Lakh</div>
          <div class="muted">Prediction generated by the Random Forest model</div>
        </div>
        """, unsafe_allow_html=True)

        st.success(
            f"{result['year']} · {result['fuel']} · {result['transmission']} · "
            f"{result['kms']:,} km driven"
        )

        with st.expander("View prediction details"):
            st.json(result)

        if st.button("Clear last result"):
            st.session_state.last_prediction = None
            st.rerun()
    else:
        st.info("Your prediction will appear here after you submit the form.")

    if st.session_state.prediction_history:
        st.markdown("### 🕘 Recent predictions")
        st.dataframe(
            pd.DataFrame(st.session_state.prediction_history),
            width="stretch",
            hide_index=True
        )

# =========================================================
# DATA ANALYSIS
# =========================================================
elif page == "📊 Data Analysis":
    page_hero(
        "📊 Data Analysis",
        "Filter your dataset and explore price patterns."
    )

    cols = st.columns(4)
    with cols[0]:
        st.metric("Records", len(df))
    with cols[1]:
        st.metric("Columns", df.shape[1])
    with cols[2]:
        st.metric("Missing values", int(df.isnull().sum().sum()))
    with cols[3]:
        st.metric("Duplicate rows", int(df.duplicated().sum()))

    years = sorted(df["Year"].dropna().unique().tolist())
    fuels = sorted(df["Fuel_Type"].dropna().astype(str).unique().tolist())
    sellers = sorted(df["Seller_Type"].dropna().astype(str).unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_years = st.multiselect("Manufacturing year", years, default=years)
    with c2:
        selected_fuels = st.multiselect("Fuel type", fuels, default=fuels)
    with c3:
        selected_sellers = st.multiselect("Seller type", sellers, default=sellers)

    filtered = df[
        df["Year"].isin(selected_years)
        & df["Fuel_Type"].astype(str).isin(selected_fuels)
        & df["Seller_Type"].astype(str).isin(selected_sellers)
    ].copy()

    search = st.text_input("Search car name")
    table_data = filtered

    if search.strip() and "Car_Name" in table_data.columns:
        table_data = table_data[
            table_data["Car_Name"].astype(str).str.contains(
                search.strip(), case=False, na=False
            )
        ]

    st.caption(f"Showing {len(table_data)} of {len(df)} records.")
    st.dataframe(table_data, width="stretch", hide_index=True)

    if filtered.empty:
        st.warning("No records match your filters.")
    else:
        chart = st.selectbox(
            "Choose a chart",
            [
                "Year vs Selling Price",
                "Kilometres vs Selling Price",
                "Selling Price by Fuel Type",
                "Selling Price by Seller Type",
                "Correlation Heatmap"
            ]
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))

        if chart == "Year vs Selling Price":
            ax.scatter(filtered["Year"], filtered["Selling_Price"], alpha=0.72)
            ax.set_xlabel("Manufacturing Year")
            ax.set_ylabel("Selling Price (Lakh)")

        elif chart == "Kilometres vs Selling Price":
            ax.scatter(filtered["Kms_Driven"], filtered["Selling_Price"], alpha=0.72)
            ax.set_xlabel("Kilometres Driven")
            ax.set_ylabel("Selling Price (Lakh)")

        elif chart == "Selling Price by Fuel Type":
            labels = sorted(filtered["Fuel_Type"].astype(str).unique())
            grouped = [
                filtered.loc[
                    filtered["Fuel_Type"].astype(str) == label,
                    "Selling_Price"
                ].dropna()
                for label in labels
            ]
            ax.boxplot(grouped, labels=labels)
            ax.set_ylabel("Selling Price (Lakh)")

        elif chart == "Selling Price by Seller Type":
            labels = sorted(filtered["Seller_Type"].astype(str).unique())
            grouped = [
                filtered.loc[
                    filtered["Seller_Type"].astype(str) == label,
                    "Selling_Price"
                ].dropna()
                for label in labels
            ]
            ax.boxplot(grouped, labels=labels)
            ax.set_ylabel("Selling Price (Lakh)")

        else:
            numeric = filtered.select_dtypes(include=np.number)
            corr = numeric.corr()
            image = ax.imshow(corr, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.index)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(chart)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📈 Model Performance":
    page_hero(
        "📈 Model Performance",
        "Evaluate predictions and inspect important features."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("🎯", "R²-based score", f"{score:.2f}%")
    with c2:
        metric_card("🎯", "Within ±10%", f"{within_10:.2f}%")
    with c3:
        metric_card("📏", "MAE", f"{mae:.3f}")
    with c4:
        metric_card("📐", "RMSE", f"{rmse:.3f}")

    st.markdown("""
    <div class="panel"><div class="panel-title">🌲 Model information</div>
    <div class="muted">
    Algorithm: Random Forest Regressor<br>
    Number of trees: 100<br>
    Train/test split: 80% / 20%<br>
    Random state: 42
    </div></div>
    """, unsafe_allow_html=True)

    view = st.radio(
        "Performance view",
        ["Actual vs Predicted", "Prediction Error", "Feature Importance"],
        horizontal=True
    )

    if view == "Actual vs Predicted":
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.scatter(y_test, predictions, alpha=0.75)
        low = min(float(y_test.min()), float(np.min(predictions)))
        high = max(float(y_test.max()), float(np.max(predictions)))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual Selling Price (Lakh)")
        ax.set_ylabel("Predicted Selling Price (Lakh)")
        ax.set_title("Actual vs Predicted Prices")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif view == "Prediction Error":
        errors = y_test.to_numpy() - predictions
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.hist(errors, bins=12, edgecolor="white")
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Actual − Predicted (Lakh)")
        ax.set_ylabel("Number of cars")
        ax.set_title("Prediction Error Distribution")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(importance["Feature"], importance["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            importance.sort_values("Importance", ascending=False).round(4),
            width="stretch",
            hide_index=True
        )

    st.info(
        "R² measures how much variation in selling price the model explains. "
        "The R²-based score is R² × 100, not classification accuracy. "
        "'Within ±10%' is the percentage of non-zero actual test prices "
        "whose predictions are within 10% of the actual price."
    )

    with st.expander("View test-set predictions"):
        results = pd.DataFrame({
            "Actual Price (Lakh)": y_test.to_numpy(),
            "Predicted Price (Lakh)": predictions,
            "Absolute Error": np.abs(y_test.to_numpy() - predictions)
        })
        st.dataframe(results.round(3), width="stretch", hide_index=True)

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":
    page_hero(
        "ℹ️ About CarPredict",
        "Project details, technologies, and workflow."
    )

    st.markdown("""
    <div class="panel"><div class="panel-title">🎯 Objective</div>
    <div class="muted">
    Build a machine-learning application that estimates used-car selling
    prices using vehicle details and provides visual analysis of the dataset.
    </div></div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel"><div class="panel-title">🛠️ Technologies</div>
        <div class="muted">
        • Python<br>• Pandas and NumPy<br>• Matplotlib<br>
        • Scikit-learn<br>• Random Forest Regression<br>• Streamlit
        </div></div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel"><div class="panel-title">📁 Dataset fields</div>
        <div class="muted">
        Car name, year, selling price, present price, kilometres driven,
        fuel type, seller type, transmission, and previous owners.
        </div></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel"><div class="panel-title">🔄 Workflow</div>
    <div class="muted">
    Dataset loading → preprocessing and one-hot encoding → train/test split →
    Random Forest training → model evaluation → interactive price prediction.
    </div></div>
    """, unsafe_allow_html=True)

    st.caption(f"Dataset loaded: {len(df)} rows and {df.shape[1]} columns.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">🚗 CarPredict · Used Car Price Prediction & Analysis · SMLDS Mini Project</div>',
    unsafe_allow_html=True
)

