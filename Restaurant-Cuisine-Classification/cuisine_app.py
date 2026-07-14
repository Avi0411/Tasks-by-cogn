import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Restaurant Cuisine Classifier",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# FIXED THEME — CONSISTENT IN LIGHT OR DARK MODE

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0E1117; }
[data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
.main { animation: fadeIn 0.8s ease-in-out; color: #E8E8E8; }
@keyframes fadeIn { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: translateY(0);} }

.hero-title {
    font-size: 2.6rem; font-weight: 800; color: #FF7A59;
    margin-bottom: 0; animation: slideIn 0.9s ease-out;
}
@keyframes slideIn { from {opacity: 0; transform: translateX(-30px);} to {opacity: 1; transform: translateX(0);} }
.hero-subtitle { font-size: 1.05rem; color: #B8BCC8; margin-top: 0.3rem; }

.kpi-card {
    background: linear-gradient(135deg, #1B1F2A, #23283A);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1rem; text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 10px 25px rgba(255,122,89,0.25); }
.kpi-value { font-size: 2.0rem; font-weight: 800; color: #FFB84D; }
.kpi-label { font-size: 0.85rem; color: #A9AEBD; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 0.2rem; }

.section-header {
    font-size: 1.6rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.8rem;
    border-left: 5px solid #FF7A59; padding-left: 0.7rem; color: #F0F0F0;
    animation: fadeIn 0.6s ease-in-out;
}

.main p, .main span, .main label, .main div { color: #E8E8E8; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #12141C, #181B26); }
section[data-testid="stSidebar"] * { color: #D7DAE3 !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #FF9E80 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] { background-color: #1F2330; border-radius: 8px; }

.stTabs [data-baseweb="tab"] { color: #C9CDD8; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #FF7A59 !important; border-bottom-color: #FF7A59 !important; }

[data-testid="stAlert"] { background-color: #1B1F2A; color: #E8E8E8; border-radius: 12px; }

.footer { text-align: center; padding: 1.5rem 0; color: #8A8F9C; font-size: 0.95rem; animation: fadeIn 1.2s ease-in-out; }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, #FF7A59, transparent); }

.prediction-box {
    background: linear-gradient(135deg, #1B1F2A, #23283A);
    border: 1px solid rgba(255,122,89,0.4);
    border-radius: 16px; padding: 1.5rem; text-align: center; margin-top: 1rem;
}
.prediction-value { font-size: 2rem; font-weight: 800; color: #FF9E80; }
</style>
""", unsafe_allow_html=True)


# LOAD DATA

@st.cache_data
def load_data():
    data = pd.read_csv("Avi.csv")
    data = data.dropna(subset=["Cuisines"])
    numeric_cols = ["Average Cost for two", "Price range", "Votes"]
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    return data

df = load_data()


# BUILD TARGET: PRIMARY CUISINE (GROUPED)

@st.cache_data
def build_target(data, top_n):
    data = data.copy()
    data["Primary Cuisine"] = data["Cuisines"].apply(lambda x: x.split(",")[0].strip())
    top_cuisines = data["Primary Cuisine"].value_counts().head(top_n).index.tolist()
    data["Cuisine Target"] = data["Primary Cuisine"].apply(
        lambda c: c if c in top_cuisines else "Other"
    )
    return data, top_cuisines


# HERO HEADER

st.markdown('<div class="hero-title">🍽️ Restaurant Cuisine Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Built by <b>Avishkar D Rathod</b> — AI/ML Enthusiastic &nbsp;|&nbsp; Classify restaurants by primary cuisine using ML 🤖</div>', unsafe_allow_html=True)
st.write("")
st.divider()


# SIDEBAR — CONTROLS

with st.sidebar:
    st.markdown("## ⚙️ Model Controls")
    st.markdown("**🤖 AI/ML Enthusiastic:** Avishkar D Rathod")
    st.markdown("**📁 Dataset:** Restaurant Dataset")

    st.divider()

    top_n = st.slider("Number of top cuisines to keep", min_value=5, max_value=25, value=15, step=1)

    model_choice = st.selectbox(
        "🧠 Select Model",
        ["Random Forest", "Logistic Regression", "Compare Both"]
    )

    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.divider()
    st.caption("Adjust these and everything below updates live ✨")


# PREPARE DATA

df_target, top_cuisines = build_target(df, top_n)

feature_cols = [
    "City", "Price range", "Average Cost for two",
    "Has Table booking", "Has Online delivery", "Votes",
    "Aggregate rating"
]

X = df_target[feature_cols].copy()
y = df_target["Cuisine Target"]

label_encoders = {}
categorical_cols = ["City", "Has Table booking", "Has Online delivery"]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
)


# TRAIN MODELS (CACHED)

@st.cache_resource
def train_model(name, X_train, y_train):
    if name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

with st.spinner("Training model(s)..."):
    if model_choice == "Compare Both":
        rf_model = train_model("Random Forest", X_train, y_train)
        lr_model = train_model("Logistic Regression", X_train, y_train)
        rf_results = evaluate(rf_model, X_test, y_test)
        lr_results = evaluate(lr_model, X_test, y_test)
    elif model_choice == "Random Forest":
        model = train_model("Random Forest", X_train, y_train)
        results = evaluate(model, X_test, y_test)
    else:
        model = train_model("Logistic Regression", X_train, y_train)
        results = evaluate(model, X_test, y_test)


# DATASET OVERVIEW

st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_target):,}</div><div class="kpi-label">Restaurants</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(top_cuisines)}</div><div class="kpi-label">Top Cuisines Kept</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{X_train.shape[0]:,}</div><div class="kpi-label">Training Rows</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{X_test.shape[0]:,}</div><div class="kpi-label">Test Rows</div></div>', unsafe_allow_html=True)

st.write("")
st.divider()


# MODEL PERFORMANCE

st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

if model_choice == "Compare Both":
    comp_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Accuracy": [rf_results["accuracy"], lr_results["accuracy"]],
        "Precision": [rf_results["precision"], lr_results["precision"]],
        "Recall": [rf_results["recall"], lr_results["recall"]],
        "F1 Score": [rf_results["f1"], lr_results["f1"]],
    })

    fig = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        template="plotly_dark", color_discrete_sequence=["#FF7A59", "#FFB84D"],
        title="Random Forest vs Logistic Regression"
    )
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp_df.set_index("Model").style.format("{:.4f}"), use_container_width=True)

    best_name = "Random Forest" if rf_results["f1"] >= lr_results["f1"] else "Logistic Regression"
    best_results = rf_results if best_name == "Random Forest" else lr_results
    best_model = rf_model if best_name == "Random Forest" else lr_model
    st.success(f"🏆 Best model based on weighted F1: **{best_name}**")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{results['accuracy']:.2%}")
    m2.metric("Precision", f"{results['precision']:.2%}")
    m3.metric("Recall", f"{results['recall']:.2%}")
    m4.metric("F1 Score", f"{results['f1']:.2%}")

    best_name = model_choice
    best_results = results
    best_model = model

st.divider()


# TABS: DETAILED ANALYSIS

st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Per-Cuisine Performance",
    "🧩 Confusion Matrix",
    "🌟 Feature Importance",
    "🔮 Try a Prediction"
])

report_dict = classification_report(
    y_test, best_results["y_pred"],
    target_names=target_encoder.classes_,
    output_dict=True, zero_division=0
)
per_class_df = pd.DataFrame(report_dict).transpose()
per_class_df = per_class_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
per_class_df = per_class_df.sort_values("f1-score", ascending=False).reset_index()
per_class_df = per_class_df.rename(columns={"index": "Cuisine"})

with tab1:
    fig = px.bar(
        per_class_df, x="Cuisine", y="f1-score",
        color="f1-score", color_continuous_scale="Sunset",
        template="plotly_dark", title=f"F1-Score by Cuisine — {best_name}",
        text_auto=".2f"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        per_class_df[["Cuisine", "precision", "recall", "f1-score", "support"]]
        .style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}),
        use_container_width=True, hide_index=True
    )

    worst = per_class_df.tail(3)["Cuisine"].tolist()
    best_c = per_class_df.head(3)["Cuisine"].tolist()

    st.info(f"""
    **Best-performing classes:** {', '.join(best_c)}

    **Weakest-performing classes:** {', '.join(worst)}

    **Likely causes of weaker performance:**
    - Class imbalance — cuisines with fewer training samples are harder to learn
    - Feature overlap — similar price range / city / vote patterns across cuisines cause confusion
    - The "Other" bucket mixes many unrelated cuisines, so its metrics can look inconsistent
    """)

with tab2:
    cm = confusion_matrix(y_test, best_results["y_pred"])
    fig = px.imshow(
        cm,
        x=target_encoder.classes_, y=target_encoder.classes_,
        color_continuous_scale="Sunset", template="plotly_dark",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"Confusion Matrix — {best_name}"
    )
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if best_name == "Random Forest":
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Sunset",
            template="plotly_dark", title="Feature Importance (Random Forest)",
            text_auto=".2f"
        )
        fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance chart is available for Random Forest. Switch model selection to view it, or choose 'Compare Both'.")
        if model_choice == "Compare Both":
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": rf_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig = px.bar(
                importance_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Sunset",
                template="plotly_dark", title="Feature Importance (Random Forest)",
                text_auto=".2f"
            )
            fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.write("Enter restaurant details to predict its primary cuisine:")

    c1, c2 = st.columns(2)
    with c1:
        input_city = st.selectbox("City", sorted(df["City"].unique()))
        input_price_range = st.slider("Price Range (1-4)", 1, 4, 2)
        input_cost = st.number_input("Average Cost for Two", min_value=0, value=500, step=50)
        input_votes = st.number_input("Votes", min_value=0, value=50, step=10)
    with c2:
        input_rating = st.slider("Aggregate Rating", 0.0, 5.0, 3.5, 0.1)
        input_table_booking = st.selectbox("Has Table Booking", ["Yes", "No"])
        input_online_delivery = st.selectbox("Has Online Delivery", ["Yes", "No"])

    if st.button("🔮 Predict Cuisine", use_container_width=True):
        try:
            input_row = pd.DataFrame([{
                "City": label_encoders["City"].transform([input_city])[0]
                        if input_city in label_encoders["City"].classes_ else 0,
                "Price range": input_price_range,
                "Average Cost for two": input_cost,
                "Has Table booking": label_encoders["Has Table booking"].transform([input_table_booking])[0],
                "Has Online delivery": label_encoders["Has Online delivery"].transform([input_online_delivery])[0],
                "Votes": input_votes,
                "Aggregate rating": input_rating
            }])

            pred_encoded = best_model.predict(input_row)[0]
            pred_label = target_encoder.inverse_transform([pred_encoded])[0]

            st.markdown(f"""
                <div class="prediction-box">
                    <div>Predicted Primary Cuisine</div>
                    <div class="prediction-value">🍽️ {pred_label}</div>
                </div>
            """, unsafe_allow_html=True)

            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_row)[0]
                proba_df = pd.DataFrame({
                    "Cuisine": target_encoder.classes_,
                    "Probability": proba
                }).sort_values("Probability", ascending=False).head(5)

                fig = px.bar(
                    proba_df, x="Probability", y="Cuisine", orientation="h",
                    color="Probability", color_continuous_scale="Sunset",
                    template="plotly_dark", title="Top 5 Predicted Probabilities",
                    text_auto=".2%"
                )
                fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()


# FOOTER

st.markdown(
    '<div class="footer"> Made with Streamlit by <b>Avishkar Rathod</b> — AI/ML Enthusiastic thankyou </div>',
    unsafe_allow_html=True
)
