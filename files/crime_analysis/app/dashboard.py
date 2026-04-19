"""
Crime Analysis and Safety Prediction System
==========================================
Full Streamlit dashboard with EDA, prediction, maps, and knowledge center.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── path setup so src/ modules are importable ────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "..", "src")
sys.path.insert(0, SRC)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocessing import full_pipeline
from models import load_artifacts
from predictor import predict_safety

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crime Analysis & Safety Prediction",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --primary: #1a1a2e;
    --accent: #e94560;
    --card-bg: #16213e;
    --text: #eaeaea;
}
.main { background-color: var(--primary); }
.stMetric { background: var(--card-bg); border-radius: 10px; padding: 10px; }
.sidebar .sidebar-content { background: var(--card-bg); }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.safe-badge   { background:#1a7a4a; color:white; padding:8px 20px; border-radius:20px; font-weight:bold; font-size:1.2em; }
.moderate-badge { background:#d68910; color:white; padding:8px 20px; border-radius:20px; font-weight:bold; font-size:1.2em; }
.risky-badge  { background:#c0392b; color:white; padding:8px 20px; border-radius:20px; font-weight:bold; font-size:1.2em; }

h1, h2, h3 { color: #e94560 !important; }
.stSelectbox label, .stSlider label, .stDateInput label { color: #eaeaea !important; }
</style>
""", unsafe_allow_html=True)


# ── cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_everything():
    data_path   = os.path.join(ROOT, "..", "data", "crime_safety_dataset.csv")
    models_dir  = os.path.join(ROOT, "..", "models")
    df, X, y, encoders, feature_cols, scaler = full_pipeline(data_path)
    model, enc, sc, fc = load_artifacts(models_dir)
    return df, model, enc, sc, fc, encoders


with st.spinner("🔄 Loading data and models…"):
    df, model, enc, sc, fc, encoders = load_everything()

CITIES   = sorted(df["city"].unique())
STATES   = sorted(df["state"].unique())
CRIMES   = sorted(df["crime_type"].unique())
LOCS     = sorted(df["location_description"].unique())
GENDERS  = sorted(df["victim_gender"].unique())
RACES    = sorted(df["victim_race"].unique())

# ── sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/police-badge.png", width=70)
st.sidebar.markdown("## 🚔 Crime Analysis System")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🗺️ Crime Map", "📈 Trends", "🔮 Safety Prediction",
     "📋 Data Explorer", "📚 Knowledge Center"],
)

# ── filters (shared) ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Global Filters")
sel_cities = st.sidebar.multiselect("City", CITIES, default=CITIES[:5])
sel_crimes = st.sidebar.multiselect("Crime Type", CRIMES, default=CRIMES)
year_range = st.sidebar.slider(
    "Year Range",
    int(df["year"].min()), int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max())),
)

# apply filters
mask = (
    df["city"].isin(sel_cities if sel_cities else CITIES) &
    df["crime_type"].isin(sel_crimes if sel_crimes else CRIMES) &
    df["year"].between(*year_range)
)
fdf = df[mask]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("🚔 Crime Analysis & Safety Prediction System")
    st.markdown("*An end-to-end ML platform for crime insights and safety forecasting*")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Incidents", f"{len(fdf):,}")
    with col2:
        st.metric("Cities Covered", fdf["city"].nunique())
    with col3:
        st.metric("Crime Types", fdf["crime_type"].nunique())
    with col4:
        risky_pct = (fdf["safety_label"] == "Risky").mean() * 100
        st.metric("Risky Cases", f"{risky_pct:.1f}%")
    with col5:
        avg_age = fdf["victim_age"].mean()
        st.metric("Avg Victim Age", f"{avg_age:.0f} yrs")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Crime type bar
    with col_a:
        ct = fdf["crime_type"].value_counts().reset_index()
        ct.columns = ["Crime Type", "Count"]
        fig = px.bar(
            ct, x="Count", y="Crime Type", orientation="h",
            color="Count", color_continuous_scale="Reds",
            title="Crime Type Distribution",
        )
        fig.update_layout(showlegend=False, height=350,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaeaea")
        st.plotly_chart(fig, use_container_width=True)

    # Safety label donut
    with col_b:
        sl = fdf["safety_label"].value_counts().reset_index()
        sl.columns = ["Label", "Count"]
        fig2 = px.pie(
            sl, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Safe":"#27ae60","Moderate":"#f39c12","Risky":"#e74c3c"},
            hole=0.45, title="Safety Level Distribution",
        )
        fig2.update_layout(height=350,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#eaeaea")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Crime by city
    with col_c:
        city_ct = fdf["city"].value_counts().head(10).reset_index()
        city_ct.columns = ["City", "Count"]
        fig3 = px.bar(city_ct, x="City", y="Count",
                      color="Count", color_continuous_scale="Blues",
                      title="Top 10 Cities by Crime Count")
        fig3.update_layout(showlegend=False, height=320,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#eaeaea", xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    # Hour distribution
    with col_d:
        fig4 = px.histogram(fdf, x="hour", nbins=24,
                            title="Crime by Hour of Day",
                            color_discrete_sequence=["#e94560"])
        fig4.update_layout(height=320, bargap=0.1,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#eaeaea")
        fig4.update_xaxes(tickvals=list(range(0,24,2)))
        st.plotly_chart(fig4, use_container_width=True)

    # Crime by gender
    gender_ct = fdf["victim_gender"].value_counts().reset_index()
    gender_ct.columns = ["Gender", "Count"]
    fig5 = px.bar(gender_ct, x="Gender", y="Count",
                  color="Gender", title="Victims by Gender",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig5.update_layout(showlegend=False, height=280,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font_color="#eaeaea")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CRIME MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Crime Map":
    st.title("🗺️ Interactive Crime Map")

    CITY_COORDS = {
        "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298), "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740), "Philadelphia": (39.9526, -75.1652),
        "San Antonio": (29.4241, -98.4936), "San Diego": (32.7157, -117.1611),
        "Dallas": (32.7767, -96.7970), "San Jose": (37.3382, -121.8863),
    }

    map_df = fdf[fdf["city"].isin(CITY_COORDS)].copy()
    map_df["lat"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (0,0))[0])
    map_df["lon"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (0,0))[1])

    # Add small jitter for visual spread
    np.random.seed(42)
    map_df["lat"] += np.random.uniform(-0.15, 0.15, len(map_df))
    map_df["lon"] += np.random.uniform(-0.15, 0.15, len(map_df))

    color_map = {"Safe": "#27ae60", "Moderate": "#f39c12", "Risky": "#e74c3c"}
    map_df["color"] = map_df["safety_label"].map(color_map)

    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon",
        color="safety_label",
        color_discrete_map=color_map,
        hover_name="city",
        hover_data={"crime_type": True, "victim_age": True,
                    "time_of_day": True, "lat": False, "lon": False},
        zoom=3.5, height=550,
        title="Crime Incidents by Safety Level",
        mapbox_style="carto-darkmatter",
    )
    fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#eaeaea",
                           legend_title="Safety Level")
    st.plotly_chart(fig_map, use_container_width=True)

    # City risk table
    st.subheader("📍 City Risk Summary")
    city_risk = (fdf.groupby("city")["safety_label"]
                   .value_counts(normalize=True)
                   .unstack(fill_value=0) * 100)
    city_risk.columns = [f"{c} %" for c in city_risk.columns]
    city_risk["Total Crimes"] = fdf.groupby("city").size()
    city_risk = city_risk.sort_values("Risky %", ascending=False).round(1)
    st.dataframe(city_risk.style.background_gradient(cmap="RdYlGn_r", subset=["Risky %"]),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.title("📈 Crime Trends & Time Analysis")

    # Monthly trend
    monthly = fdf.groupby(["year","month"]).size().reset_index(name="count")
    monthly["period"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
    fig_trend = px.line(monthly, x="period", y="count", markers=True,
                        title="Monthly Crime Trend",
                        color_discrete_sequence=["#e94560"])
    fig_trend.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                             paper_bgcolor="rgba(0,0,0,0)", font_color="#eaeaea")
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Heatmap: day of week vs hour
        heat = fdf.groupby(["day_of_week","hour"]).size().reset_index(name="count")
        heat_pivot = heat.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
        heat_pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig_heat = px.imshow(heat_pivot, color_continuous_scale="YlOrRd",
                             title="Crime Intensity: Day vs Hour",
                             labels={"x":"Hour","y":"Day","color":"Count"})
        fig_heat.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                                font_color="#eaeaea")
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Crime type over years
        yearly_type = fdf.groupby(["year","crime_type"]).size().reset_index(name="count")
        fig_yt = px.bar(yearly_type, x="year", y="count", color="crime_type",
                        barmode="stack", title="Crime Types Over Years",
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig_yt.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#eaeaea")
        st.plotly_chart(fig_yt, use_container_width=True)

    # Weekend vs weekday
    fdf_copy = fdf.copy()
    fdf_copy["week_type"] = fdf_copy["is_weekend"].map({0:"Weekday",1:"Weekend"})
    ww = fdf_copy.groupby(["crime_type","week_type"]).size().reset_index(name="count")
    fig_ww = px.bar(ww, x="crime_type", y="count", color="week_type",
                    barmode="group", title="Weekday vs Weekend Crime Pattern",
                    color_discrete_map={"Weekday":"#3498db","Weekend":"#e74c3c"})
    fig_ww.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#eaeaea",
                          xaxis_tickangle=-30)
    st.plotly_chart(fig_ww, use_container_width=True)

    # Time of day breakdown
    tod_ct = fdf["time_of_day"].value_counts().reset_index()
    tod_ct.columns = ["Time of Day","Count"]
    order = ["Night","Morning","Afternoon","Evening","Late Night"]
    tod_ct["Time of Day"] = pd.Categorical(tod_ct["Time of Day"], categories=order, ordered=True)
    tod_ct = tod_ct.sort_values("Time of Day")
    fig_tod = px.bar(tod_ct, x="Time of Day", y="Count",
                     color="Count", color_continuous_scale="Reds",
                     title="Crime by Time of Day")
    fig_tod.update_layout(height=300, showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#eaeaea")
    st.plotly_chart(fig_tod, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SAFETY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Safety Prediction":
    st.title("🔮 Real-Time Safety Prediction")
    st.markdown("Enter details below to predict the safety level of an area/incident.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📍 Location")
            city     = st.selectbox("City", CITIES)
            state    = st.selectbox("State", STATES)
            location = st.selectbox("Location Type", LOCS)

        with col2:
            st.markdown("#### 🕐 Time & Date")
            date_in  = st.date_input("Date")
            time_in  = st.time_input("Time")
            crime_type = st.selectbox("Crime Type", CRIMES)

        with col3:
            st.markdown("#### 👤 Victim Info")
            victim_age    = st.slider("Victim Age", 1, 90, 28)
            victim_gender = st.selectbox("Gender", GENDERS)
            victim_race   = st.selectbox("Race", RACES)

        submitted = st.form_submit_button("🔍 Predict Safety Level", use_container_width=True)

    if submitted:
        result = predict_safety(
            model, enc, sc, fc,
            city, state, location, crime_type,
            victim_gender, victim_race, victim_age,
            str(date_in), time_in.strftime("%H:%M"),
        )
        st.markdown("---")
        label = result["label"]
        badge_cls = {"Safe":"safe-badge","Moderate":"moderate-badge","Risky":"risky-badge"}.get(label,"")
        st.markdown(f"### Prediction Result: <span class='{badge_cls}'>{result['icon']} {label}</span>",
                    unsafe_allow_html=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Safety Level", f"{result['icon']} {label}")
        with col_r2:
            st.metric("Severity Score", f"{result['severity_score']}/5")
        with col_r3:
            st.metric("Time of Day", result["time_of_day"])
        with col_r4:
            st.metric("Weekend?", "Yes" if result["is_weekend"] else "No")

        # Probability bar chart
        proba = result["probabilities"]
        prob_df = pd.DataFrame(list(proba.items()), columns=["Label","Probability %"])
        color_m = {"Safe":"#27ae60","Moderate":"#f39c12","Risky":"#e74c3c"}
        fig_p = px.bar(prob_df, x="Label", y="Probability %",
                       color="Label", color_discrete_map=color_m,
                       title="Prediction Confidence")
        fig_p.update_layout(showlegend=False, height=280,
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             font_color="#eaeaea", yaxis_range=[0,100])
        st.plotly_chart(fig_p, use_container_width=True)

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Safety Recommendations")
        recs = {
            "Safe":     ["✅ Area is generally safe.", "🚶 Walk confidently but stay alert.",
                         "📱 Keep emergency contacts handy."],
            "Moderate": ["⚠️ Exercise caution in this area.", "👥 Stay in groups if possible.",
                         "💡 Avoid poorly lit streets at night.", "📲 Share your location with someone you trust."],
            "Risky":    ["🚨 High-risk area – avoid if possible.", "🏠 Stay indoors after dark.",
                         "📞 Keep police hotline saved (911).", "🚗 Use trusted transportation only.",
                         "📸 Document surroundings if needed for report."],
        }
        for rec in recs.get(label, []):
            st.markdown(f"- {rec}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")

    st.subheader("Raw Dataset (filtered)")
    display_cols = ["date","time","city","state","crime_type",
                    "location_description","victim_age","victim_gender",
                    "victim_race","safety_label","severity_score","time_of_day"]
    st.dataframe(fdf[display_cols].reset_index(drop=True), use_container_width=True, height=400)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(fdf[["victim_age","severity_score","hour","month","year"]]
                     .describe().round(2), use_container_width=True)

    with col2:
        st.subheader("🔢 Crime Type Counts")
        st.dataframe(fdf["crime_type"].value_counts().reset_index()
                     .rename(columns={"crime_type":"Crime Type","count":"Count"}),
                     use_container_width=True)

    st.subheader("📉 Model Performance Comparison")
    import json
    report_path = os.path.join(ROOT, "..", "reports", "model_comparison.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            mc = json.load(f)
        mc_df = pd.DataFrame(mc).T.reset_index().rename(columns={"index":"Model"})
        st.dataframe(mc_df, use_container_width=True)
        fig_mc = px.bar(mc_df, x="Model", y=["accuracy","precision","recall","f1"],
                        barmode="group", title="Model Comparison",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_mc.update_layout(height=320, yaxis_range=[0,1.05],
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#eaeaea")
        st.plotly_chart(fig_mc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KNOWLEDGE CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Knowledge Center":
    st.title("📚 Knowledge Center – Crime Awareness & Safety")

    tabs = st.tabs(["🔎 Crime Types", "🛡️ Safety Tips", "📞 Emergency Contacts",
                    "📝 How to Report"])

    with tabs[0]:
        st.subheader("Understanding Different Crime Types")
        crimes_info = {
            "🔪 Homicide": "Intentional killing of a person. The most severe violent crime. Report immediately to police (911).",
            "🏠 Burglary": "Unlawful entry into a structure to commit a crime. Secure doors/windows; install alarms.",
            "💰 Robbery": "Taking property by force or threat. Do not resist; cooperate and report afterward.",
            "🔥 Arson": "Deliberately setting fire to property. Call 911 immediately; do not enter burning buildings.",
            "👊 Assault": "Physical attack or threat of violence. Document injuries; seek medical care; file a police report.",
            "💔 Domestic Violence": "Abuse by an intimate partner or family member. Call 1-800-799-SAFE for support.",
            "💊 Drug Offense": "Illegal possession, distribution, or manufacturing of controlled substances.",
            "🛒 Theft": "Taking someone's property without permission. Report to police and insurance.",
            "🎨 Vandalism": "Intentional damage to property. Photograph damage and report to local authorities.",
            "💳 Fraud": "Deceptive acts for financial gain. Contact the FTC at ReportFraud.ftc.gov.",
        }
        for crime, desc in crimes_info.items():
            with st.expander(crime):
                st.write(desc)

    with tabs[1]:
        st.subheader("🛡️ Safety Tips & Precautions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**🏡 At Home**
- Lock all doors and windows at night
- Install a doorbell camera or security system
- Do not open doors to strangers
- Keep exterior areas well-lit
- Know your neighbors

**🚶 On the Street**
- Stay in well-lit, populated areas
- Be aware of your surroundings
- Avoid using phone/headphones in isolated areas
- Trust your instincts — leave if uncomfortable
- Walk confidently and with purpose
""")
        with col2:
            st.markdown("""
**🚗 While Driving**
- Lock car doors and keep windows up
- Park in well-lit, busy areas
- Check backseat before entering
- Don't leave valuables visible
- Keep your phone charged

**💻 Online Safety**
- Use strong, unique passwords
- Enable two-factor authentication
- Do not share personal details publicly
- Be wary of phishing emails
- Report online fraud immediately
""")

    with tabs[2]:
        st.subheader("📞 Emergency Contacts")
        contacts = {
            "🚨 Police Emergency": "911",
            "📞 Non-Emergency Police": "311",
            "🔥 Fire Department": "911",
            "🏥 Ambulance": "911",
            "💔 Domestic Violence Hotline": "1-800-799-7233 (SAFE)",
            "🧠 Mental Health Crisis Line": "988",
            "🕵️ FBI Tips": "1-800-CALL-FBI",
            "💳 FTC Fraud Hotline": "1-877-382-4357",
            "🏠 Runaway Hotline": "1-800-786-2929",
            "💊 Substance Abuse Helpline": "1-800-662-4357",
        }
        for name, number in contacts.items():
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"**{name}**")
            with col2:
                st.code(number)

    with tabs[3]:
        st.subheader("📝 How and Where to Report Crimes")
        st.markdown("""
### Reporting a Crime in the US

#### 🚨 Emergency (crime in progress)
Call **911** immediately. Provide:
- Your location (address or landmarks)
- Nature of the emergency
- Description of suspects if known

#### 📋 Non-Emergency Reporting
1. **Visit your local police station** – Bring any evidence or documentation
2. **Call non-emergency line (311)** – For non-urgent incidents
3. **Online reporting portals** – Many cities have web forms for minor crimes

#### 🌐 Federal Crimes & Specialized Agencies
| Crime | Where to Report |
|-------|----------------|
| Internet crime / fraud | [IC3.gov](https://ic3.gov) |
| Identity theft | [IdentityTheft.gov](https://identitytheft.gov) |
| FTC fraud | [ReportFraud.ftc.gov](https://reportfraud.ftc.gov) |
| Drug trafficking | DEA – [dea.gov/tips](https://www.dea.gov/tips) |
| Civil rights violations | FBI – [tips.fbi.gov](https://tips.fbi.gov) |

#### 📌 What to Document
- Date, time, and location of incident
- Description of suspect(s): height, clothing, vehicle
- Names of witnesses
- Photos or videos if safely possible
- Any injury or property damage details
        """)

# ── footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
🔒 Crime Analysis & Safety Prediction System<br>
Built with Python · Streamlit · Scikit-learn · Plotly
</small>
""", unsafe_allow_html=True)
