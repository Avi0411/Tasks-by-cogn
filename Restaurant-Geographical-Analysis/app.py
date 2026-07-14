import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent

# ====================================================
# PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="Restaurant Geographical Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# FIXED THEME — LOOKS THE SAME IN LIGHT OR DARK MODE
# ====================================================
st.markdown("""
<style>

/* ---- Force a consistent app background regardless of system theme ---- */
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.main {
    animation: fadeIn 0.8s ease-in-out;
    color: #E8E8E8;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* ---- Title ---- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #FF7A59;
    margin-bottom: 0;
    animation: slideIn 0.9s ease-out;
}
@keyframes slideIn {
    from {opacity: 0; transform: translateX(-30px);}
    to {opacity: 1; transform: translateX(0);}
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #B8BCC8;
    margin-top: 0.3rem;
}

/* ---- KPI cards: fixed dark card, always high-contrast ---- */
.kpi-card {
    background: linear-gradient(135deg, #1B1F2A, #23283A);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.kpi-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 25px rgba(255,122,89,0.25);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #FFB84D;
}
.kpi-label {
    font-size: 0.9rem;
    color: #A9AEBD;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 5px solid #FF7A59;
    padding-left: 0.7rem;
    color: #F0F0F0;
    animation: fadeIn 0.6s ease-in-out;
}

/* ---- Force readable text everywhere inside main content ---- */
.main p, .main span, .main label, .main div {
    color: #E8E8E8;
}

/* ---- Sidebar: fixed dark navy, always consistent ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141C, #181B26);
}
section[data-testid="stSidebar"] * {
    color: #D7DAE3 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FF9E80 !important;
}

/* ---- Selectbox / inputs inside sidebar ---- */
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #1F2330;
    border-radius: 8px;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] {
    color: #C9CDD8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #FF7A59 !important;
    border-bottom-color: #FF7A59 !important;
}

/* ---- Dataframe / expander text ---- */
.streamlit-expanderHeader {
    color: #E8E8E8 !important;
}

/* ---- Info box ---- */
[data-testid="stAlert"] {
    background-color: #1B1F2A;
    color: #E8E8E8;
    border-radius: 12px;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 1.5rem 0;
    color: #8A8F9C;
    font-size: 0.95rem;
    animation: fadeIn 1.2s ease-in-out;
}

/* ---- Divider glow ---- */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #FF7A59, transparent);
}

</style>
""", unsafe_allow_html=True)


# LOADING DATASET

@st.cache_data
def load_data():
    csv_path = BASE_DIR / "Avi.csv"
    st.write("CSV Path:", csv_path)
    st.write("CSV Exists:", csv_path.exists())
    data = pd.read_csv(csv_path)

    df = load_data()
    return data



# ====================================================
# HERO HEADER
# ====================================================
st.markdown('<div class="hero-title">🌍 Restaurant Geographical Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Built by <b>Avishkar D Rathod</b> — AI/ML Enthusiastic &nbsp;|&nbsp; Analyze restaurant locations, ratings, cuisines and price ranges 📊</div>', unsafe_allow_html=True)
st.write("")
st.divider()

# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:
    st.markdown("## 📊 Dashboard Controls")
    st.markdown("**🤖 Built by :** Avishkar D Rathod")
    st.markdown("**📁 Dataset:** Restaurant Dataset")

    st.markdown("**🛠️ Tech Stack**")
    st.markdown("- Python 🐍\n- Streamlit ⚡\n- Folium 🗺️\n- Plotly 📈\n- Pandas 🐼")

    st.divider()

    selected_city = st.selectbox(
        "🏙️ Select City",
        ["All"] + sorted(df["City"].dropna().unique().tolist())
    )

    st.divider()
    st.caption("Tip: switch cities to update every chart and the map live ✨")

# ====================================================
# FILTER DATA
# ====================================================
if selected_city == "All":
    filtered_df = df
else:
    filtered_df = df[df["City"] == selected_city]

# ====================================================
# KPI CARDS
# ====================================================
st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{len(filtered_df):,}</div>
            <div class="kpi-label">Restaurants</div>
        </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{filtered_df['City'].nunique()}</div>
            <div class="kpi-label">Cities</div>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{filtered_df['Locality'].nunique()}</div>
            <div class="kpi-label">Localities</div>
        </div>
    """, unsafe_allow_html=True)

with k4:
    avg_rating = filtered_df["Aggregate rating"].mean()
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">⭐ {avg_rating:.1f}</div>
            <div class="kpi-label">Avg Rating</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# ====================================================
# MAP
# ====================================================
st.markdown('<div class="section-header">🗺️ Restaurant Locations</div>', unsafe_allow_html=True)

with st.spinner("Rendering map..."):
    restaurant_map = folium.Map(
        location=[
            filtered_df["Latitude"].mean(),
            filtered_df["Longitude"].mean()
        ],
        zoom_start=11,
        tiles="CartoDB dark_matter"
    )

    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color="#FF7A59",
            fill=True,
            fill_color="#FFB84D",
            fill_opacity=0.85,
            popup=folium.Popup(
                f"""
                <b>{row['Restaurant Name']}</b><br>
                ⭐ Rating: {row['Aggregate rating']}<br>
                🍽️ {row['Cuisines']}
                """,
                max_width=250
            )
        ).add_to(restaurant_map)

    st_folium(restaurant_map, width=1200, height=520)

st.divider()

# ====================================================
# TABBED ANALYTICS SECTION
# ====================================================
st.markdown('<div class="section-header">📈 Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏙️ Top Cities",
    "📍 Top Localities",
    "⭐ Ratings",
    "💰 Price Range",
    "🍕 Cuisines"
])

color_scale = px.colors.sequential.Sunset

# Common dark-plot template so charts also stay consistent regardless of theme
plot_template = "plotly_dark"

with tab1:
    city_count = df["City"].value_counts().head(10).reset_index()
    city_count.columns = ["City", "Restaurants"]

    fig = px.bar(
        city_count, x="City", y="Restaurants",
        color="Restaurants", color_continuous_scale=color_scale,
        text="Restaurants", title="Top 10 Cities by Number of Restaurants",
        template=plot_template
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(transition_duration=500, showlegend=False,
                       paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    locality_count = filtered_df["Locality"].value_counts().head(10).reset_index()
    locality_count.columns = ["Locality", "Restaurants"]

    fig = px.bar(
        locality_count, x="Locality", y="Restaurants",
        color="Restaurants", color_continuous_scale=color_scale,
        text="Restaurants", title="Top 10 Localities",
        template=plot_template
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(transition_duration=500, showlegend=False,
                       paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    rating = (
        df.groupby("City")["Aggregate rating"]
          .mean()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    fig = px.bar(
        rating, x="City", y="Aggregate rating",
        color="Aggregate rating", color_continuous_scale="Viridis",
        text_auto=".2f", title="Average Rating by City (Top 10)",
        template=plot_template
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(transition_duration=500, showlegend=False,
                       paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    price = (
        df.groupby("City")["Price range"]
          .mean()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )
    fig = px.bar(
        price, x="City", y="Price range",
        color="Price range", color_continuous_scale="Magma",
        text_auto=".2f", title="Average Price Range by City (Top 10)",
        template=plot_template
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(transition_duration=500, showlegend=False,
                       paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    popular = (
        df.groupby("City")["Cuisines"]
          .apply(lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "Unknown")
          .reset_index(name="Most Popular Cuisine")
    )
    st.dataframe(
        popular,
        use_container_width=True,
        hide_index=True
    )

st.divider()

# ====================================================
# INSIGHTS
# ====================================================
st.markdown('<div class="section-header">📌 Key Insights</div>', unsafe_allow_html=True)

st.info("""
- Restaurants are concentrated in a handful of major cities.
- Cities with more restaurants generally offer greater cuisine diversity.
- Average ratings vary meaningfully across cities.
- Premium price ranges are more common in metropolitan cities.
- Explore exact locations and details interactively using the map above.
""")

# ====================================================
# RAW DATA (COLLAPSIBLE)
# ====================================================
with st.expander("📄 Show Raw Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

# ====================================================
# FOOTER
# ====================================================
st.divider()
st.markdown(
    '<div class="footer">❤️ Made with Streamlit by <b>Avishkar Rathod</b> — AI/ML Engineer ❤️</div>',
    unsafe_allow_html=True
)
