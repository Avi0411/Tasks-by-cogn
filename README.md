# Cognifyz Internship — Machine Learning Tasks

A collection of 3 Machine Learning tasks completed as part of the **Cognifyz Technologies** internship program, using restaurant data to build predictive models, classifiers, and geospatial visualizations — each with an interactive Streamlit dashboard.

## Live Dashboards

| Task | Live App |
|---|---|
| Restaurant Rating Prediction | [Launch dashboard](https://your-app-url.streamlit.app) *(update once deployed)* |
| Restaurant Cuisine Classification | [Launch dashboard](https://your-app-url.streamlit.app) *(update once deployed)* |
| Restaurant Geographical Analysis | [Launch dashboard](https://your-app-url.streamlit.app) *(update once deployed)* |

## Tasks

### 1. Restaurant Rating Prediction
An ML regression model trained to predict restaurant ratings based on cost, location, and cuisine features.

**Folder:** `Restaurant-Rating-Prediction/`
- `Notebook.ipynb` — full ML workflow (data prep, training, evaluation)
- `app.py` — Streamlit dashboard
- `mlmodel.pkl` — saved trained model
- `Scaler.pkl` — saved feature scaler
- `Avi.csv` — dataset used
- `requirements.txt` — dependencies

### 2. Restaurant Cuisine Classification
An ML classification model to predict restaurant cuisine type from text and categorical features.

**Folder:** `Restaurant-Cuisine-Classification/`
- `cuisine.ipynb` — full ML workflow (data prep, training, evaluation)
- `cuisine_app.py` — Streamlit dashboard
- `Avi.csv` — dataset used
- `requirements.txt` — dependencies

### 3. Restaurant Geographical Analysis
Analyzes and visualizes restaurant locations and trends, including an interactive map.

**Folder:** `Restaurant-Geographical-Analysis/`
- `geographical_analysis.ipynb` — full analysis workflow
- `app.py` — Streamlit dashboard
- `restaurant_map.html` — interactive map visualization
- `Avi.csv` — dataset used
- `requirements.txt` — dependencies

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- Jupyter Notebook

## Run Locally

Each task folder is self-contained. To run a task's dashboard on your own machine:

```bash
cd Restaurant-Rating-Prediction
pip install -r requirements.txt
streamlit run app.py
```

Repeat with the corresponding folder and file for the other two tasks (`Restaurant-Cuisine-Classification/cuisine_app.py` and `Restaurant-Geographical-Analysis/app.py`).

## Project Structure

```
Cog-Tasks/
├── Restaurant-Rating-Prediction/
│   ├── Notebook.ipynb
│   ├── app.py
│   ├── mlmodel.pkl
│   ├── Scaler.pkl
│   ├── Avi.csv
│   └── requirements.txt
├── Restaurant-Cuisine-Classification/
│   ├── cuisine.ipynb
│   ├── cuisine_app.py
│   ├── Avi.csv
│   └── requirements.txt
└── Restaurant-Geographical-Analysis/
    ├── geographical_analysis.ipynb
    ├── app.py
    ├── restaurant_map.html
    ├── Avi.csv
    └── requirements.txt
```

## License

This project is for educational purposes as part of the Cognifyz Technologies internship program.
