from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Spotify.csv"
PUBLIC_DIR = BASE_DIR.parent / "public"
OUTPUT_FILE = PUBLIC_DIR / "index.html"

GREEN = "#1ed760"
MINT = "#67e8a5"
AMBER = "#f4b95a"
BLUE = "#72b7f8"
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
CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["artist", "song", "duration_ms", "year", "popularity"])
    df["genre"] = df["genre"].fillna("Unknown")
    df["primary_genre"] = df["genre"].apply(lambda value: str(value).split(",")[0].strip())
    df["duration_min"] = df["duration_ms"] / 60000
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10
    df["explicit"] = df["explicit"].astype(bool)
    return df


def style_chart(fig, title, height=430, legend=True):
    fig.update_layout(
        template="plotly_dark",
        title=title,
        title_font_color="#f4f7f5",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,17,14,0.78)",
        font=dict(family="Manrope, Arial, sans-serif", color="#dce5df"),
        margin=dict(l=32, r=24, t=64, b=42),
        height=height,
        hoverlabel=dict(bgcolor="#101612", bordercolor=GREEN, font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def chart_html(fig):
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=CHART_CONFIG,
        default_width="100%",
    )


def build_charts(df):
    charts = []

    fig_popularity = px.histogram(
        df,
        x="popularity",
        nbins=30,
        color_discrete_sequence=[GREEN],
    )
    fig_popularity.update_xaxes(title_text="Popularity score")
    fig_popularity.update_yaxes(title_text="Tracks")
    charts.append(("Popularity distribution", chart_html(style_chart(fig_popularity, "Popularity distribution"))))

    yearly_pop = df.groupby("year")["popularity"].agg(["mean", "median"]).reset_index()
    fig_year = go.Figure()
    fig_year.add_trace(
        go.Scatter(
            x=yearly_pop["year"],
            y=yearly_pop["mean"],
            mode="lines+markers",
            name="Average",
            line=dict(color=GREEN, width=3),
        )
    )
    fig_year.add_trace(
        go.Scatter(
            x=yearly_pop["year"],
            y=yearly_pop["median"],
            mode="lines+markers",
            name="Median",
            line=dict(color=AMBER, width=2, dash="dash"),
        )
    )
    fig_year.update_xaxes(title_text="Year")
    fig_year.update_yaxes(title_text="Popularity score")
    charts.append(("Popularity over time", chart_html(style_chart(fig_year, "Popularity over time"))))

    corr_matrix = df[MODEL_FEATURES].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="BrBG",
    )
    charts.append(("Audio feature correlations", chart_html(style_chart(fig_corr, "Audio feature correlations", height=620))))

    genre_counts = df["primary_genre"].value_counts().head(15).sort_values()
    fig_genres = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation="h",
        color=genre_counts.values,
        color_continuous_scale=["#2b332f", MINT],
    )
    fig_genres.update_xaxes(title_text="Tracks")
    fig_genres.update_yaxes(title_text="")
    charts.append(("Top genres", chart_html(style_chart(fig_genres, "Top genres by track count", height=540, legend=False))))

    artist_counts = df["artist"].value_counts().head(15).sort_values()
    fig_artists = px.bar(
        x=artist_counts.values,
        y=artist_counts.index,
        orientation="h",
        color=artist_counts.values,
        color_continuous_scale=["#2b332f", GREEN],
    )
    fig_artists.update_xaxes(title_text="Tracks")
    fig_artists.update_yaxes(title_text="")
    charts.append(("Top artists", chart_html(style_chart(fig_artists, "Top artists by track count", height=540, legend=False))))

    yearly_counts = df.groupby("year").size().reset_index(name="track_count")
    yearly_popularity = df.groupby("year")["popularity"].mean().reset_index()
    yearly_data = yearly_counts.merge(yearly_popularity, on="year")
    fig_release = make_subplots(specs=[[{"secondary_y": True}]])
    fig_release.add_trace(
        go.Scatter(
            x=yearly_data["year"],
            y=yearly_data["track_count"],
            mode="lines+markers",
            name="Track count",
            line=dict(color=GREEN, width=3),
        ),
        secondary_y=False,
    )
    fig_release.add_trace(
        go.Scatter(
            x=yearly_data["year"],
            y=yearly_data["popularity"],
            mode="lines+markers",
            name="Avg popularity",
            line=dict(color=BLUE, width=2),
        ),
        secondary_y=True,
    )
    fig_release.update_xaxes(title_text="Year")
    fig_release.update_yaxes(title_text="Tracks", secondary_y=False)
    fig_release.update_yaxes(title_text="Avg popularity", secondary_y=True)
    charts.append(("Release trends", chart_html(style_chart(fig_release, "Release volume and popularity"))))

    explicit_counts = df["explicit"].value_counts().reindex([False, True], fill_value=0)
    fig_explicit = px.pie(
        values=explicit_counts.values,
        names=["Non-explicit", "Explicit"],
        hole=0.58,
        color_discrete_sequence=[GREEN, AMBER],
    )
    charts.append(("Explicit tracks", chart_html(style_chart(fig_explicit, "Explicit vs non-explicit tracks"))))

    return charts


def metric(label, value):
    return f"""
    <article class="metric">
        <span>{label}</span>
        <strong>{value}</strong>
    </article>
    """


def build_html(df):
    top_genre = df["primary_genre"].mode().iat[0]
    metrics = "\n".join(
        [
            metric("Total tracks", f"{len(df):,}"),
            metric("Unique artists", f"{df['artist'].nunique():,}"),
            metric("Average popularity", f"{df['popularity'].mean():.1f}"),
            metric("Median duration", f"{df['duration_min'].median():.1f} min"),
            metric("Top genre", top_genre),
            metric("Year range", f"{df['year'].min()}-{df['year'].max()}"),
        ]
    )
    chart_sections = "\n".join(
        f"""
        <section class="chart-block" id="{title.lower().replace(' ', '-')}">
            {html}
        </section>
        """
        for title, html in build_charts(df)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Spotify Music Analytics Dashboard</title>
    <meta name="description" content="A Spotify analytics dashboard built from a 2,000-track dataset.">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            color-scheme: dark;
            --green: #1ed760;
            --mint: #67e8a5;
            --ink: #f4f7f5;
            --muted: #9ba7a0;
            --panel: rgba(20, 24, 23, 0.88);
            --border: rgba(255, 255, 255, 0.09);
        }}

        * {{
            box-sizing: border-box;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            margin: 0;
            color: var(--ink);
            font-family: "Manrope", system-ui, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(30, 215, 96, 0.16), transparent 34rem),
                radial-gradient(circle at bottom right, rgba(244, 185, 90, 0.09), transparent 30rem),
                linear-gradient(135deg, #07100c 0%, #0f1512 48%, #111613 100%);
        }}

        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.2;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, black 0%, transparent 82%);
        }}

        .shell {{
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 44px 0 64px;
            position: relative;
        }}

        .hero {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: clamp(22px, 4vw, 42px);
            background:
                linear-gradient(135deg, rgba(30, 215, 96, 0.14), rgba(103, 232, 165, 0.03)),
                rgba(16, 20, 18, 0.82);
            box-shadow: 0 20px 70px rgba(0, 0, 0, 0.26);
        }}

        .eyebrow {{
            color: var(--mint);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        h1 {{
            max-width: 820px;
            margin: 0;
            font-size: clamp(2.4rem, 7vw, 5.6rem);
            line-height: 0.95;
            letter-spacing: 0;
            text-wrap: balance;
        }}

        .intro {{
            max-width: 68ch;
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.7;
            margin: 1.1rem 0 0;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 22px;
        }}

        .metric {{
            min-height: 104px;
            border-radius: 8px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--border);
        }}

        .metric span {{
            display: block;
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .metric strong {{
            display: block;
            margin-top: 8px;
            font-size: clamp(1.35rem, 2.5vw, 2rem);
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }}

        .nav {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0 6px;
        }}

        .nav a {{
            color: var(--ink);
            text-decoration: none;
            border: 1px solid var(--border);
            border-radius: 7px;
            padding: 9px 12px;
            background: rgba(255, 255, 255, 0.035);
            transition: background 180ms ease, transform 180ms ease;
        }}

        .nav a:hover {{
            background: rgba(30, 215, 96, 0.15);
            transform: translateY(-1px);
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}

        .chart-block {{
            min-width: 0;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            background: var(--panel);
        }}

        .chart-block:nth-child(3) {{
            grid-column: 1 / -1;
        }}

        footer {{
            color: var(--muted);
            margin-top: 28px;
            line-height: 1.6;
        }}

        @media (max-width: 860px) {{
            .metrics,
            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            .chart-block:nth-child(3) {{
                grid-column: auto;
            }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="eyebrow">Spotify catalog analytics</div>
            <h1>Turn 2,000 tracks into listening intelligence.</h1>
            <p class="intro">
                Explore popularity, release timing, audio traits, genre behavior, and explicit-track patterns from a compact Spotify dataset.
                This Vercel build is a static dashboard generated from the same data used by the Streamlit app.
            </p>
            <div class="metrics">{metrics}</div>
        </section>
        <nav class="nav" aria-label="Dashboard sections">
            <a href="#popularity-distribution">Popularity</a>
            <a href="#popularity-over-time">Trends</a>
            <a href="#audio-feature-correlations">Correlations</a>
            <a href="#top-genres">Genres</a>
            <a href="#top-artists">Artists</a>
            <a href="#explicit-tracks">Explicit</a>
        </nav>
        <section class="chart-grid">{chart_sections}</section>
        <footer>
            Interactive filtering and clustering are available in the local Streamlit app. Run <code>streamlit run dash.py</code> for the full exploratory version.
        </footer>
    </main>
</body>
</html>
"""


def main():
    PUBLIC_DIR.mkdir(exist_ok=True)
    df = load_data()
    OUTPUT_FILE.write_text(build_html(df), encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
