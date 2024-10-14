import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="CTR Forecasting", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("ctr.csv")
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    return data

data = load_data()

# Réinitialiser l'index
data.reset_index(inplace=True)

# Affichage des premières lignes
st.write("### Aperçu des données")
st.write(data.head())

# Visualisation des clics et impressions au fil du temps
st.write("### Clicks and Impressions Over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Clicks'], mode='lines', name='Clicks'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Impressions'], mode='lines', name='Impressions'))
fig.update_layout(title='Clicks and Impressions Over Time')
st.plotly_chart(fig)

# Calcul du taux de clic (CTR)
data['CTR'] = (data['Clicks'] / data['Impressions']) * 100
st.write("### Click-Through Rate (CTR) Over Time")
fig_ctr = px.line(data, x=data['Date'], y='CTR', title='Click-Through Rate (CTR) Over Time')
st.plotly_chart(fig_ctr)

# Analyse exploratoire des données par jour de la semaine
data['DayOfWeek'] = data['Date'].dt.dayofweek
day_of_week_ctr = data.groupby('DayOfWeek')['CTR'].mean().reset_index()
day_of_week_ctr['DayOfWeek'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

st.write("### CTR moyen par jour de la semaine")
fig_dow = px.bar(day_of_week_ctr, x='DayOfWeek', y='CTR', title='Average CTR by Day of the Week')
st.plotly_chart(fig_dow)

# Préparation à la modélisation SARIMAX
st.write("### Prévision du CTR à l'aide de SARIMAX")

# Série temporelle
time_series = data.set_index('Date')['CTR']

# Différenciation
differenced_series = time_series.diff().dropna()

# Tracer ACF et PACF de la série temporelle différenciée
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
st.pyplot(fig)

# Modélisation SARIMAX
p, d, q, s = 1, 1, 1, 12
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(p, d, q, s))
results = model.fit()

# Affichage du résumé du modèle
st.write("#### Résumé du modèle")
st.text(results.summary())

# Prédictions futures
future_steps = 100  # Nombre de périodes à prédire directement
predictions = results.predict(len(time_series), len(time_series) + future_steps - 1)

# Création d'un DataFrame avec les données originales et les prédictions
forecast = pd.DataFrame({'Predictions': predictions})

# Créer un nouvel index pour le DataFrame des prévisions
forecast.index = pd.date_range(start=time_series.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')

# Visualisation des prédictions
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast['Predictions'], mode='lines', name='Predictions'))
fig_forecast.add_trace(go.Scatter(x=time_series.index, y=time_series, mode='lines', name='Original Data'))
fig_forecast.update_layout(title='CTR Forecasting', xaxis_title='Time Period', yaxis_title='CTR')
st.plotly_chart(fig_forecast)
