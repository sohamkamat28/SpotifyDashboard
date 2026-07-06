# Spotify Music Analytics Dashboard

A Streamlit dashboard for exploring a 2,000-track Spotify dataset across popularity, release timing, genres, audio features, and KMeans song clusters.

## What is inside

- Portable dataset loading from `Spotify.csv`
- Sidebar filters for year, popularity, genre, artist, explicit tracks, and cluster count
- Plotly visualizations for distribution, trends, correlations, genre behavior, and PCA-reduced clusters
- Polished dark Spotify-inspired UI with responsive summary cards and empty states

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dash.py
```

The app expects `Spotify.csv` to live in the same folder as `dash.py`.
