# GEOScouter

GEOScouter helps you profile and compare public datasets on [GEO](https://www.ncbi.nlm.nih.gov/geo/) from an exported `gds_result.txt` search.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

1. Run a search on [GEO DataSets](https://www.ncbi.nlm.nih.gov/gds/), export **gds_result.txt**
2. Upload the file in the app and run the pipeline
3. **Filter** by platform, sample count, and (optionally) sample metadata **before** plotting
4. Visualize, build a GSE comparison list, export tables

## Project layout

```
geoscouter/          # reusable library (pipeline, filters, plots)
streamlit_app.py     # Streamlit UI entry point
data/input/          # example gds_result.txt
legacy/app/          # previous monolithic scripts (reference)
```

## Deploy on Streamlit Community Cloud

1. Push the `dev` branch to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → repository `Mmaycon/GEOScouter`, branch `dev`
4. Main file path: **`streamlit_app.py`**
5. Deploy

No Chrome/Selenium is required on Streamlit Cloud; supplementary files are parsed via SOFT and HTTP.

## Development branch

Active rework lives on **`dev`**: modular package, upstream filtering, improved filename similarity, and in-app help text.

## License

MIT — see [LICENSE](LICENSE).
