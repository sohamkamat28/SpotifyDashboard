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
pip install -r streamlit_app/requirements-local.txt
streamlit run streamlit_app/dash.py
```

The Streamlit app and CSV live in `streamlit_app/`.

## Deploy on Vercel

Vercel does not run Streamlit as a long-lived interactive server. This repo includes a committed static dashboard in `public/index.html`.

If you change the CSV or charts, regenerate it locally:

```bash
pip install -r streamlit_app/requirements-local.txt
python streamlit_app/build_static.py
```

Vercel uses `vercel.json` to serve the `public` folder directly, with no Python function bundle.
