"""
streamlit_app.py — Hotel Recommendation Web Application
MLOps Capstone Project

Run locally:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
    streamlit run streamlit_app.py

GitHub: https://github.com/<your-username>/mlops-travel-capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Hotel Recommender",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .rec-rank { font-size: 2rem; font-weight: bold; opacity: 0.8; }
    .rec-name { font-size: 1.2rem; font-weight: bold; }
    .rec-score { font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hotels=pd.read_csv("../data/hotels.csv")
    users  =pd.read_csv("../data/users.csv")
    return hotels, users


@st.cache_resource
def load_model_artefacts():
    svd_model    = joblib.load("../models/svd_model.pkl")
    matrix       = joblib.load("../models/interaction_matrix.pkl")
    recon        = joblib.load("../models/reconstructed_matrix.pkl")
    hotel_names  = joblib.load("../models/hotel_names.pkl")
    return svd_model, matrix, recon, hotel_names


# ── Recommendation function ───────────────────────────────────────────────────
def recommend_hotels(user_code, matrix, recon, top_n=3):
    if user_code not in matrix.index:
        popular = matrix.sum(axis=0).sort_values(ascending=False).head(top_n)
        return [{"hotel": h, "score": round(float(s), 2), "note": "cold start"}
                for h, s in popular.items()]
    visited   = set(matrix.columns[matrix.loc[user_code] > 0])
    scores    = recon.loc[user_code]
    unvisited = scores.drop(list(visited), errors="ignore")
    if len(unvisited) == 0:
        recs = scores.sort_values(ascending=False).head(top_n)
    else:
        recs = unvisited.sort_values(ascending=False).head(top_n)
    return [{"hotel": h, "score": round(float(s), 2), "note": "unvisited"}
            for h, s in recs.items()]


# ── Load everything ───────────────────────────────────────────────────────────
try:
    hotels, users = load_data()
    svd_model, matrix, recon, hotel_names = load_model_artefacts()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing_file = str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hotel.png", width=64)
    st.title("🏨 Hotel Recommender")
    st.caption("MLOps Capstone — Travel & Tourism")
    st.markdown("---")

    if data_loaded:
        # User selector
        all_users = sorted(matrix.index.tolist())
        selected_user = st.selectbox(
            "Select a User",
            options=all_users,
            index=0,
            help="Choose a userCode to generate personalised hotel recommendations"
        )
        top_n = st.slider("Number of recommendations", min_value=1, max_value=9, value=3)
        st.markdown("---")

        # User info
        user_info = users[users["code"] == selected_user]
        if not user_info.empty:
            u = user_info.iloc[0]
            st.markdown(f"**User:** {selected_user}")
            st.markdown(f"**Company:** {u.get('company', 'N/A')}")
            st.markdown(f"**Age:** {u.get('age', 'N/A')}")
            st.markdown(f"**Gender:** {u.get('gender', 'N/A')}")

        st.markdown("---")
        # SVD model info
        st.markdown("**Model Info**")
        st.markdown(f"Components: `{svd_model.n_components}`")
        var_exp = svd_model.explained_variance_ratio_.sum()
        st.progress(float(var_exp), text=f"Variance explained: {var_exp*100:.1f}%")
    else:
        st.error(f"Could not load files.\n\n{missing_file}\n\nRun Notebook 3 first to generate model artefacts.")
        st.stop()


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏨 Travel Hotel Recommendation System")
st.caption("Collaborative Filtering with SVD · MLOps Capstone Project")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations",
    "📊 EDA & Insights",
    "🔬 Model Analysis",
    "📋 Dataset Explorer"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Recommendations
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Top {top_n} Hotel Recommendations for User {selected_user}")

    recs = recommend_hotels(selected_user, matrix, recon, top_n)
    visited_hotels = set(matrix.columns[matrix.loc[selected_user] > 0]) if selected_user in matrix.index else set()

    col_recs, col_profile = st.columns([3, 2])

    with col_recs:
        for rank, rec in enumerate(recs, 1):
            score_pct = max(0, min(100, (rec["score"] / 1000) * 100))
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-name">🏩 {rec['hotel']}</div>
                <div class="rec-score">Preference score: {rec['score']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        if recs:
            # Score bar chart
            st.markdown("#### Recommendation Score Comparison")
            fig, ax = plt.subplots(figsize=(7, 3))
            hotel_names_r = [r["hotel"] for r in recs]
            scores = [r["score"] for r in recs]
            colors = ["#667eea", "#764ba2", "#f093fb"][:len(recs)]
            bars = ax.barh(hotel_names_r[::-1], scores[::-1], color=colors[::-1], edgecolor="white")
            for bar, score in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{score:.1f}", va="center", fontsize=10)
            ax.set_xlabel("Predicted Preference Score")
            ax.set_title("Recommendation Scores")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_profile:
        st.markdown("#### User Booking Profile")

        # Hotels visited vs not visited
        visited_list = sorted(visited_hotels)
        not_visited  = sorted(set(hotel_names) - visited_hotels)

        st.markdown(f"**Visited ({len(visited_list)}):**")
        for h in visited_list:
            spend = matrix.loc[selected_user, h] if selected_user in matrix.index else 0
            st.markdown(f"✅ {h} — ₹{spend:,.0f} total spend")

        st.markdown(f"**Not yet visited ({len(not_visited)}):**")
        for h in not_visited:
            st.markdown(f"⬜ {h}")

        # Spend breakdown
        if selected_user in matrix.index:
            user_spends = matrix.loc[selected_user]
            if user_spends.sum() > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                user_spends[user_spends > 0].sort_values(ascending=True).plot(
                    kind="barh", ax=ax2, color="#667eea", edgecolor="white")
                ax2.set_title("Total Spend per Hotel", fontsize=10)
                ax2.set_xlabel("Total Spend (₹)")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: EDA & Insights
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis — Hotels Dataset")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Bookings",    f"{len(hotels):,}")
    m2.metric("Unique Users",      f"{hotels['userCode'].nunique():,}")
    m3.metric("Hotels",            f"{hotels['name'].nunique()}")
    m4.metric("Avg Spend / Stay",  f"₹{hotels['total'].mean():,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Hotel visit frequency
        fig, ax = plt.subplots(figsize=(7, 4))
        hotel_counts = hotels["name"].value_counts()
        hotel_counts.plot(kind="bar", ax=ax, color="#667eea", edgecolor="white")
        ax.set_title("Total Bookings per Hotel", fontweight="bold")
        ax.set_xlabel("Hotel")
        ax.set_ylabel("Bookings")
        ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Price per night
        fig, ax = plt.subplots(figsize=(7, 4))
        hotels["price"].hist(bins=30, ax=ax, color="#764ba2", edgecolor="white", alpha=0.85)
        ax.axvline(hotels["price"].mean(), color="red", linestyle="--",
                   label=f"Mean: ₹{hotels['price'].mean():.0f}")
        ax.set_title("Price per Night Distribution", fontweight="bold")
        ax.set_xlabel("Price per Night (₹)")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        # Avg total spend by hotel
        fig, ax = plt.subplots(figsize=(7, 4))
        avg_spend = hotels.groupby("name")["total"].mean().sort_values(ascending=True)
        avg_spend.plot(kind="barh", ax=ax, color="#f093fb", edgecolor="white")
        ax.set_title("Average Total Spend by Hotel", fontweight="bold")
        ax.set_xlabel("Avg Total Spend (₹)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stay duration
        fig, ax = plt.subplots(figsize=(7, 4))
        hotels["days"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="#667eea", edgecolor="white")
        ax.set_title("Stay Duration Distribution", fontweight="bold")
        ax.set_xlabel("Days")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Interaction matrix heatmap
    st.markdown("#### User-Hotel Interaction Matrix (first 50 users)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_matrix = matrix.head(50)
    sns.heatmap(sample_matrix, cmap="YlOrRd", ax=ax, linewidths=0.1,
                cbar_kws={"label": "Total Spend (₹)"})
    ax.set_title("User × Hotel Interaction Matrix — Total Spend", fontweight="bold")
    ax.set_xlabel("Hotel")
    ax.set_ylabel("User Code")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Analysis
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("SVD Model Analysis")

    c1, c2 = st.columns(2)

    with c1:
        # Explained variance per component
        st.markdown("#### Variance Explained per SVD Component")
        fig, ax = plt.subplots(figsize=(6, 4))
        components = range(1, svd_model.n_components + 1)
        ax.bar(components, svd_model.explained_variance_ratio_,
               color="#667eea", edgecolor="white")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"Total explained: {svd_model.explained_variance_ratio_.sum()*100:.1f}%")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Component 1 vs 2 user scatter
        st.markdown("#### User Distribution in Latent Space (Component 1 vs 2)")
        from sklearn.preprocessing import normalize as sk_normalize
        matrix_norm = sk_normalize(matrix.values, norm="l2")
        user_factors = svd_model.transform(matrix_norm)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(user_factors[:, 0], user_factors[:, 1],
                   alpha=0.4, s=10, color="#764ba2")
        if selected_user in matrix.index:
            idx = matrix.index.get_loc(selected_user)
            ax.scatter(user_factors[idx, 0], user_factors[idx, 1],
                       s=120, color="red", zorder=5, label=f"User {selected_user}")
            ax.legend()
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Users in SVD Latent Space")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        # Hotel loadings on each component
        st.markdown("#### Hotel Loadings per SVD Component")
        item_factors_df = pd.DataFrame(
            svd_model.components_.T,
            index=hotel_names,
            columns=[f"Component {i+1}" for i in range(svd_model.n_components)]
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(item_factors_df, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.3, annot_kws={"size": 8})
        ax.set_title("Hotel Loadings per SVD Component")
        ax.set_xlabel("SVD Component")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Model metrics summary
        st.markdown("#### Model Performance Metrics")
        metrics_data = {
            "Metric": ["n_components", "Explained Variance", "Hotels covered", "Sparsity"],
            "Value": [
                svd_model.n_components,
                f"{svd_model.explained_variance_ratio_.sum()*100:.1f}%",
                f"{len(hotel_names)} / {len(hotel_names)} (100%)",
                f"{(matrix == 0).sum().sum() / matrix.size * 100:.1f}%"
            ]
        }
        st.table(pd.DataFrame(metrics_data))

        # Predicted score heatmap for current user's row
        if selected_user in recon.index:
            st.markdown(f"#### Predicted Scores for User {selected_user}")
            user_pred = recon.loc[selected_user].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = ["#f093fb" if h in visited_hotels else "#667eea"
                      for h in user_pred.index]
            ax.bar(user_pred.index, user_pred.values, color=colors, edgecolor="white")
            ax.set_title("Predicted preference score per hotel")
            ax.set_xlabel("Hotel")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor="#f093fb", label="Already visited"),
                                Patch(facecolor="#667eea", label="Not yet visited")]
            ax.legend(handles=legend_elements, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Dataset Explorer
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Dataset Explorer")

    sub1, sub2 = st.tabs(["Hotels", "Users"])

    with sub1:
        st.markdown(f"**{len(hotels):,} rows · {len(hotels.columns)} columns**")
        st.dataframe(hotels.head(100), use_container_width=True)
        st.markdown("**Descriptive statistics:**")
        st.dataframe(hotels.describe(), use_container_width=True)

    with sub2:
        st.markdown(f"**{len(users):,} rows · {len(users.columns)} columns**")
        st.dataframe(users.head(100), use_container_width=True)
        st.markdown("**Gender distribution:**")
        gender_counts = users["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_counts["Percentage"] = (gender_counts["Count"] / len(users) * 100).round(1)
        st.dataframe(gender_counts, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MLOps Capstone Project · Travel & Tourism · Hotel Recommendation System")
