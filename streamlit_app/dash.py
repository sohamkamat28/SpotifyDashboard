from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Spotify.csv"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
MODEL_FEATURES = AUDIO_FEATURES + ["tempo", "loudness"]
GREEN = "#1ed760"
MINT = "#67e8a5"
AMBER = "#f4b95a"
BLUE = "#72b7f8"
SURFACE = "#141817"
PLOT_TEMPLATE = "plotly_dark"
CHART_CONFIG = {"displayModeBar": False, "displaylogo": False}


st.set_page_config(
    page_title="Spotify Music Analytics Dashboard",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --spotify-green: #1ed760;
            --spotify-mint: #67e8a5;
            --ink: #f4f7f5;
            --muted: #9ba7a0;
            --panel: rgba(20, 24, 23, 0.88);
            --panel-strong: #161c19;
            --border: rgba(255, 255, 255, 0.08);
        }

        html {
            scroll-behavior: smooth;
        }

        .stApp {
            color: var(--ink);
            font-family: "Manrope", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(30, 215, 96, 0.16), transparent 34rem),
                radial-gradient(circle at bottom right, rgba(244, 185, 90, 0.09), transparent 30rem),
                linear-gradient(135deg, #07100c 0%, #0f1512 48%, #111613 100%);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            opacity: 0.22;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, black 0%, transparent 80%);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2.2rem;
            padding-bottom: 3.5rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            color: rgba(244, 247, 245, 0.72);
        }

        h1, h2, h3, h4, h5, h6, .stMarkdown {
            font-family: "Manrope", system-ui, sans-serif;
        }

        h3 {
            font-weight: 800;
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: rgba(7, 13, 10, 0.92);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ink);
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: rgba(244, 247, 245, 0.78);
        }

        .dashboard-hero {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: clamp(1.15rem, 2.5vw, 2rem);
            margin-bottom: 1.4rem;
            background:
                linear-gradient(135deg, rgba(30, 215, 96, 0.14), rgba(103, 232, 165, 0.03)),
                rgba(16, 20, 18, 0.82);
            box-shadow: 0 20px 70px rgba(0, 0, 0, 0.26);
        }

        .eyebrow {
            color: var(--spotify-mint);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 5vw, 4.4rem);
            line-height: 0.95;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0;
            text-wrap: balance;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 68ch;
            font-size: 1rem;
            line-height: 1.65;
            margin: 1rem 0 0;
        }

        .hero-stat-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.35rem;
        }

        .hero-stat {
            min-height: 94px;
            border-radius: 8px;
            padding: 0.9rem;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .hero-stat span {
            display: block;
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-stat strong {
            display: block;
            color: var(--ink);
            font-family: "Manrope", system-ui, sans-serif;
            font-size: clamp(1.2rem, 2.2vw, 1.8rem);
            font-weight: 800;
            margin-top: 0.35rem;
            font-variant-numeric: tabular-nums;
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            border-radius: 8px;
            padding: 1rem;
            background: var(--panel);
            border: 1px solid var(--border);
            min-height: 114px;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: "Manrope", system-ui, sans-serif;
            font-size: clamp(1.35rem, 2.1vw, 2rem);
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            letter-spacing: 0;
            white-space: normal;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            min-height: 42px;
            padding: 0.45rem 0.78rem;
            border-radius: 7px 7px 0 0;
            color: var(--muted);
            font-size: 0.91rem;
            transition: background 180ms ease, color 180ms ease;
        }

        .stTabs [aria-selected="true"] {
            color: #07100c;
            background: var(--spotify-green);
            font-weight: 800;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid rgba(30, 215, 96, 0.45);
            background: rgba(30, 215, 96, 0.11);
            color: var(--ink);
            font-weight: 700;
            transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--spotify-green);
            background: rgba(30, 215, 96, 0.2);
            transform: translateY(-1px);
        }

        .stButton > button:active,
        .stDownloadButton > button:active {
            transform: translateY(0) scale(0.99);
        }

        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .empty-state {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            background: rgba(20, 24, 23, 0.86);
            color: var(--muted);
        }

        @media (max-width: 900px) {
            .hero-stat-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .hero-stat-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_chart_layout(fig, title=None, height=None, legend=True):
    """Give every Plotly figure the same dashboard treatment."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        title_font_color="#f4f7f5",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,17,14,0.55)",
        font=dict(family="Manrope, sans-serif", color="#dce5df"),
        margin=dict(l=24, r=24, t=62 if title else 24, b=38),
        hoverlabel=dict(bgcolor="#101612", bordercolor=GREEN, font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=legend,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.11)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.07)",
        zerolinecolor="rgba(255,255,255,0.11)",
    )
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_data
def load_data():
    """Load and preprocess the Spotify dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Could not find the dataset at {DATA_FILE}.")
        return None
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        return None

    df = df.dropna(subset=["artist", "song", "duration_ms", "year", "popularity"])
    df["genre"] = df["genre"].fillna("Unknown")
    df["primary_genre"] = df["genre"].apply(
        lambda value: str(value).split(",")[0].strip() if pd.notna(value) else "Unknown"
    )
    df["duration_min"] = df["duration_ms"] / 60000
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10
    df["explicit"] = df["explicit"].astype(bool)
    return df


def create_filters(df):
    """Create sidebar controls and return selected values."""
    st.sidebar.markdown("## Listening lens")
    st.sidebar.caption("Narrow the catalog without losing the full dashboard context.")

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_range = st.sidebar.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    genres = sorted(df["primary_genre"].dropna().unique())
    selected_genres = st.sidebar.multiselect(
        "Genres",
        genres,
        default=[],
        placeholder="All genres",
    )

    top_artists = df["artist"].value_counts().head(150).index.tolist()
    selected_artists = st.sidebar.multiselect(
        "Artists",
        sorted(top_artists),
        default=[],
        placeholder="All artists",
        help="Showing the 150 most common artists keeps the control fast.",
    )

    popularity_range = st.sidebar.slider(
        "Popularity score",
        min_value=int(df["popularity"].min()),
        max_value=int(df["popularity"].max()),
        value=(int(df["popularity"].min()), int(df["popularity"].max())),
    )

    explicit_filter = st.sidebar.radio(
        "Explicit tracks",
        ["All", "Non-explicit", "Explicit only"],
        horizontal=True,
    )

    k_clusters = st.sidebar.slider(
        "Cluster count",
        min_value=2,
        max_value=8,
        value=3,
        help="Used in the clustering tab.",
    )

    return {
        "year_range": year_range,
        "selected_genres": selected_genres,
        "selected_artists": selected_artists,
        "popularity_range": popularity_range,
        "explicit_filter": explicit_filter,
        "k_clusters": k_clusters,
    }


def apply_filters(df, filters):
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["year"] >= filters["year_range"][0])
        & (filtered_df["year"] <= filters["year_range"][1])
        & (filtered_df["popularity"] >= filters["popularity_range"][0])
        & (filtered_df["popularity"] <= filters["popularity_range"][1])
    ]
    if filters["selected_genres"]:
        filtered_df = filtered_df[
            filtered_df["primary_genre"].isin(filters["selected_genres"])
        ]
    if filters["selected_artists"]:
        filtered_df = filtered_df[filtered_df["artist"].isin(filters["selected_artists"])]
    if filters["explicit_filter"] == "Explicit only":
        filtered_df = filtered_df[filtered_df["explicit"]]
    elif filters["explicit_filter"] == "Non-explicit":
        filtered_df = filtered_df[~filtered_df["explicit"]]
    return filtered_df


def render_hero(df, filtered_df):
    top_genre = filtered_df["primary_genre"].mode().iat[0] if not filtered_df.empty else "None"
    avg_pop = filtered_df["popularity"].mean() if not filtered_df.empty else 0
    year_span = (
        f"{filtered_df['year'].min()}-{filtered_df['year'].max()}"
        if not filtered_df.empty
        else "No matches"
    )

    st.markdown(
        f"""
        <section class="dashboard-hero">
            <div class="eyebrow">Spotify catalog analytics</div>
            <h1 class="hero-title">Turn 2,000 tracks into listening intelligence.</h1>
            <p class="hero-copy">
                Explore popularity, release timing, audio traits, genre behavior, and KMeans
                song clusters from a compact Spotify dataset.
            </p>
            <div class="hero-stat-row">
                <div class="hero-stat"><span>Filtered tracks</span><strong>{len(filtered_df):,}</strong></div>
                <div class="hero-stat"><span>Total catalog</span><strong>{len(df):,}</strong></div>
                <div class="hero-stat"><span>Average popularity</span><strong>{avg_pop:.1f}</strong></div>
                <div class="hero-stat"><span>Top genre</span><strong>{top_genre}</strong></div>
            </div>
            <p class="hero-copy">Current window: {year_span}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <h3>No tracks match these filters</h3>
            <p>Broaden the year, popularity, genre, artist, or explicit-track filters to bring the catalog back.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_overview_tab(df):
    st.markdown("### Catalog overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total tracks", f"{len(df):,}")
    with col2:
        st.metric("Unique artists", f"{df['artist'].nunique():,}")
    with col3:
        st.metric("Avg popularity", f"{df['popularity'].mean():.1f}")
    with col4:
        st.metric("Median duration", f"{df['duration_min'].median():.1f} min")
    with col5:
        top_genre = df["primary_genre"].mode().iat[0]
        st.metric("Top genre", top_genre)

    col1, col2 = st.columns([0.9, 1.1])
    with col1:
        st.markdown("#### Dataset pulse")
        stats_df = pd.DataFrame(
            {
                "Metric": [
                    "Year range",
                    "Explicit tracks",
                    "Average energy",
                    "Average danceability",
                    "Average valence",
                ],
                "Value": [
                    f"{df['year'].min()} - {df['year'].max()}",
                    f"{df['explicit'].sum():,} ({df['explicit'].mean() * 100:.1f}%)",
                    f"{df['energy'].mean():.3f}",
                    f"{df['danceability'].mean():.3f}",
                    f"{df['valence'].mean():.3f}",
                ],
            }
        )
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Audio feature summary")
        summary_stats = df[AUDIO_FEATURES].describe().round(3)
        st.dataframe(summary_stats, use_container_width=True)


def create_popularity_insights_tab(df):
    st.markdown("### Popularity analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df,
            x="popularity",
            nbins=30,
            color_discrete_sequence=[GREEN],
        )
        apply_chart_layout(
            fig_hist,
            title="Popularity distribution",
            height=430,
            legend=False,
        )
        fig_hist.update_xaxes(title_text="Popularity score")
        fig_hist.update_yaxes(title_text="Number of tracks")
        st.plotly_chart(fig_hist, use_container_width=True, config=CHART_CONFIG)

    with col2:
        fig_box = px.box(df, y="popularity", points="outliers", color_discrete_sequence=[MINT])
        apply_chart_layout(fig_box, title="Popularity spread", height=430, legend=False)
        fig_box.update_yaxes(title_text="Popularity score")
        st.plotly_chart(fig_box, use_container_width=True, config=CHART_CONFIG)

    yearly_pop = df.groupby("year")["popularity"].agg(["mean", "median", "count"]).reset_index()
    fig_year = go.Figure()
    fig_year.add_trace(
        go.Scatter(
            x=yearly_pop["year"],
            y=yearly_pop["mean"],
            mode="lines+markers",
            name="Average",
            line=dict(color=GREEN, width=3),
            hovertemplate="Year: %{x}<br>Avg popularity: %{y:.1f}<extra></extra>",
        )
    )
    fig_year.add_trace(
        go.Scatter(
            x=yearly_pop["year"],
            y=yearly_pop["median"],
            mode="lines+markers",
            name="Median",
            line=dict(color=AMBER, width=2, dash="dash"),
            hovertemplate="Year: %{x}<br>Median popularity: %{y:.1f}<extra></extra>",
        )
    )
    apply_chart_layout(fig_year, title="Popularity trends over time", height=470)
    fig_year.update_xaxes(title_text="Year")
    fig_year.update_yaxes(title_text="Popularity score")
    st.plotly_chart(fig_year, use_container_width=True, config=CHART_CONFIG)


def create_correlation_tab(df):
    st.markdown("### Audio feature correlations")
    corr_matrix = df[MODEL_FEATURES].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="BrBG",
    )
    apply_chart_layout(fig_heatmap, title="Audio feature correlation matrix", height=620)
    fig_heatmap.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    st.plotly_chart(fig_heatmap, use_container_width=True, config=CHART_CONFIG)

    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append(
                {
                    "Feature 1": corr_matrix.columns[i],
                    "Feature 2": corr_matrix.columns[j],
                    "Correlation": corr_matrix.iloc[i, j],
                }
            )
    corr_pairs_df = pd.DataFrame(corr_pairs)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Strongest positive relationships")
        st.dataframe(corr_pairs_df.nlargest(5, "Correlation").round(3), hide_index=True)
    with col2:
        st.markdown("#### Strongest negative relationships")
        st.dataframe(corr_pairs_df.nsmallest(5, "Correlation").round(3), hide_index=True)


def create_feature_vs_popularity_tab(df):
    st.markdown("### Feature analysis")
    selected_feature = st.selectbox("Audio feature", MODEL_FEATURES, index=0)
    sample_df = df.sample(min(1000, len(df)), random_state=42).copy()

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(
            sample_df,
            x=selected_feature,
            y="popularity",
            hover_name="song",
            hover_data={"artist": True, "year": True, selected_feature: ":.3f"},
            trendline="ols",
            color_discrete_sequence=[GREEN],
        )
        apply_chart_layout(
            fig_scatter,
            title=f"Popularity vs {selected_feature.title()}",
            height=500,
            legend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config=CHART_CONFIG)

    with col2:
        binned_df = df.copy()
        binned_df["popularity_bin"] = pd.cut(
            binned_df["popularity"],
            bins=[-1, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Very high"],
        )
        fig_box = px.box(
            binned_df.dropna(subset=["popularity_bin"]),
            x="popularity_bin",
            y=selected_feature,
            color="popularity_bin",
            color_discrete_sequence=[AMBER, BLUE, MINT, GREEN],
        )
        apply_chart_layout(
            fig_box,
            title=f"{selected_feature.title()} by popularity level",
            height=500,
            legend=False,
        )
        fig_box.update_xaxes(title_text="Popularity level")
        fig_box.update_yaxes(title_text=selected_feature.title())
        st.plotly_chart(fig_box, use_container_width=True, config=CHART_CONFIG)

    correlation = df[selected_feature].corr(df["popularity"])
    st.info(
        f"Correlation between {selected_feature.title()} and popularity: **{correlation:.3f}**"
    )


def create_release_trends_tab(df):
    st.markdown("### Release trends")
    yearly_counts = df.groupby("year").size().reset_index(name="track_count")
    yearly_popularity = df.groupby("year")["popularity"].mean().reset_index()
    yearly_data = yearly_counts.merge(yearly_popularity, on="year")

    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trends.add_trace(
        go.Scatter(
            x=yearly_data["year"],
            y=yearly_data["track_count"],
            mode="lines+markers",
            name="Track count",
            line=dict(color=GREEN, width=3),
            hovertemplate="Year: %{x}<br>Tracks: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_trends.add_trace(
        go.Scatter(
            x=yearly_data["year"],
            y=yearly_data["popularity"],
            mode="lines+markers",
            name="Avg popularity",
            line=dict(color=AMBER, width=2),
            hovertemplate="Year: %{x}<br>Avg popularity: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    apply_chart_layout(fig_trends, title="Release volume and popularity", height=480)
    fig_trends.update_xaxes(title_text="Year")
    fig_trends.update_yaxes(title_text="Number of tracks", secondary_y=False)
    fig_trends.update_yaxes(title_text="Average popularity", secondary_y=True)
    st.plotly_chart(fig_trends, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        decade_stats = df.groupby("decade").agg(
            {
                "song": "count",
                "popularity": "mean",
                "energy": "mean",
                "danceability": "mean",
            }
        ).round(2)
        decade_stats.columns = [
            "Track count",
            "Avg popularity",
            "Avg energy",
            "Avg danceability",
        ]
        st.markdown("#### Stats by decade")
        st.dataframe(decade_stats, use_container_width=True)

    with col2:
        st.markdown("#### Top genre release patterns")
        top_genres = df["primary_genre"].value_counts().head(5).index
        genre_year_data = []
        for genre in top_genres:
            genre_df = df[df["primary_genre"] == genre]
            genre_counts = genre_df.groupby("year").size()
            for year, count in genre_counts.items():
                genre_year_data.append({"year": year, "genre": genre, "count": count})

        if genre_year_data:
            genre_plot_df = pd.DataFrame(genre_year_data)
            fig_genre_trends = px.line(
                genre_plot_df,
                x="year",
                y="count",
                color="genre",
                color_discrete_sequence=[GREEN, AMBER, BLUE, MINT, "#f78fb3"],
            )
            apply_chart_layout(fig_genre_trends, title="Top 5 genres over time", height=395)
            fig_genre_trends.update_yaxes(title_text="Tracks")
            st.plotly_chart(fig_genre_trends, use_container_width=True, config=CHART_CONFIG)


def create_top_artists_genres_tab(df):
    st.markdown("### Top artists and genres")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 15 artists by track count")
        artist_counts = df["artist"].value_counts().head(15).sort_values()
        fig_artists = px.bar(
            x=artist_counts.values,
            y=artist_counts.index,
            orientation="h",
            color=artist_counts.values,
            color_continuous_scale=["#2b332f", GREEN],
        )
        apply_chart_layout(fig_artists, title="Most represented artists", height=540, legend=False)
        fig_artists.update_xaxes(title_text="Tracks")
        fig_artists.update_yaxes(title_text="")
        st.plotly_chart(fig_artists, use_container_width=True, config=CHART_CONFIG)

    with col2:
        st.markdown("#### Top 15 genres by track count")
        genre_counts = df["primary_genre"].value_counts().head(15).sort_values()
        fig_genres = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation="h",
            color=genre_counts.values,
            color_continuous_scale=["#2b332f", MINT],
        )
        apply_chart_layout(fig_genres, title="Most represented genres", height=540, legend=False)
        fig_genres.update_xaxes(title_text="Tracks")
        fig_genres.update_yaxes(title_text="")
        st.plotly_chart(fig_genres, use_container_width=True, config=CHART_CONFIG)


def create_audio_features_by_genre_tab(df):
    st.markdown("### Audio features by genre")
    selected_feature = st.selectbox("Feature", AUDIO_FEATURES, index=0)
    feature_genre = (
        df.groupby("primary_genre")[selected_feature]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .head(20)
        .sort_values(selected_feature)
    )
    fig_feature = px.bar(
        feature_genre,
        x=selected_feature,
        y="primary_genre",
        orientation="h",
        color=selected_feature,
        color_continuous_scale=["#27322d", GREEN],
    )
    apply_chart_layout(
        fig_feature,
        title=f"Average {selected_feature.title()} by genre",
        height=640,
        legend=False,
    )
    fig_feature.update_xaxes(title_text=selected_feature.title())
    fig_feature.update_yaxes(title_text="")
    st.plotly_chart(fig_feature, use_container_width=True, config=CHART_CONFIG)


def create_clustering_tab(df, k_clusters=3):
    st.markdown("### Track clustering")
    if len(df) < k_clusters:
        st.warning("Not enough tracks for the selected number of clusters.")
        return

    st.markdown(
        "KMeans groups tracks using standardized audio features, then PCA compresses the result into two dimensions for the plot."
    )

    cluster_df = df.copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(cluster_df[MODEL_FEATURES])

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    cluster_df["Cluster"] = kmeans.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(x_scaled)
    cluster_df["PC1"] = components[:, 0]
    cluster_df["PC2"] = components[:, 1]

    sample_df = cluster_df.sample(min(1000, len(cluster_df)), random_state=42)
    fig_cluster = px.scatter(
        sample_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_name="song",
        hover_data=["artist", "primary_genre", "popularity"],
        color_continuous_scale=["#2b332f", GREEN],
    )
    apply_chart_layout(fig_cluster, title="Track clusters, PCA reduced", height=560)
    st.plotly_chart(fig_cluster, use_container_width=True, config=CHART_CONFIG)

    st.markdown("#### Cluster summary")
    cluster_summary = (
        cluster_df.groupby("Cluster")
        .agg(
            Track_Count=("song", "count"),
            Avg_Popularity=("popularity", "mean"),
            Avg_Danceability=("danceability", "mean"),
            Avg_Energy=("energy", "mean"),
            Avg_Valence=("valence", "mean"),
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(cluster_summary, hide_index=True, use_container_width=True)


def create_outliers_explicit_tab(df):
    st.markdown("### Outliers and explicit tracks")
    col1, col2 = st.columns([0.8, 1.2])
    with col1:
        explicit_counts = df["explicit"].value_counts().reindex([False, True], fill_value=0)
        fig_explicit = px.pie(
            values=explicit_counts.values,
            names=["Non-explicit", "Explicit"],
            hole=0.58,
            color_discrete_sequence=[GREEN, AMBER],
        )
        apply_chart_layout(
            fig_explicit,
            title="Explicit vs non-explicit tracks",
            height=440,
            legend=True,
        )
        st.plotly_chart(fig_explicit, use_container_width=True, config=CHART_CONFIG)

    with col2:
        q1 = df["popularity"].quantile(0.25)
        q3 = df["popularity"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df["popularity"] < lower_bound) | (df["popularity"] > upper_bound)]
        st.metric("Popularity outliers", f"{len(outliers):,}")
        st.caption(
            f"Outside the IQR bounds of {lower_bound:.1f} to {upper_bound:.1f} popularity points."
        )
        if not outliers.empty:
            st.dataframe(
                outliers[["song", "artist", "popularity", "year", "primary_genre"]]
                .sort_values("popularity", ascending=False),
                hide_index=True,
                use_container_width=True,
            )


def main():
    df = load_data()
    if df is None:
        st.stop()

    filters = create_filters(df)
    filtered_df = apply_filters(df, filters)

    st.sidebar.markdown("---")
    st.sidebar.metric("Tracks in view", f"{len(filtered_df):,}")
    st.sidebar.metric("Artists in view", f"{filtered_df['artist'].nunique():,}")

    render_hero(df, filtered_df)

    if filtered_df.empty:
        render_empty_state()
        st.stop()

    tab_names = [
        "Overview",
        "Popularity",
        "Correlations",
        "Feature analysis",
        "Release trends",
        "Artists and genres",
        "Genre features",
        "Clustering",
        "Outliers",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        create_overview_tab(filtered_df)
    with tabs[1]:
        create_popularity_insights_tab(filtered_df)
    with tabs[2]:
        create_correlation_tab(filtered_df)
    with tabs[3]:
        create_feature_vs_popularity_tab(filtered_df)
    with tabs[4]:
        create_release_trends_tab(filtered_df)
    with tabs[5]:
        create_top_artists_genres_tab(filtered_df)
    with tabs[6]:
        create_audio_features_by_genre_tab(filtered_df)
    with tabs[7]:
        create_clustering_tab(filtered_df, filters["k_clusters"])
    with tabs[8]:
        create_outliers_explicit_tab(filtered_df)


if __name__ == "__main__":
    main()
