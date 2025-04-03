import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, t
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Cálculo de Value-at-Risk (VaR) y Expected Shortfall (ES)")

#######################################---BACKEND---##################################################
def ajustar_t(rendimientos):
    """Ajusta una distribución t-Student usando Maximum Likelihood Estimation (MLE)"""
    def neg_log_likelihood(params):
        df, loc, scale = params
        return -np.sum(t.logpdf(rendimientos, df=df, loc=loc, scale=scale))
    
    initial_guess = [3, np.mean(rendimientos), np.std(rendimientos)]
    bounds = [(2.1, 100), (-1, 1), (0.001, 1)]
    result = minimize(neg_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x[0]

@st.cache_data
def obtener_datos(stocks):
    return yf.download(stocks, start="2010-01-01")['Close']

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

#######################################---FRONTEND---##################################################
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
df_precios = obtener_datos(stocks_lista)
df_rendimientos = calcular_rendimientos(df_precios)

stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)
if not stock_seleccionado:
    st.stop()

# ============================ PARTE (b): Métricas estadísticas ======================================
rendimientos = df_rendimientos[stock_seleccionado]
st.subheader("📊 Métricas Estadísticas")
col1, col2, col3 = st.columns(3)
col1.metric("Media Diaria", f"{rendimientos.mean():.4%}")
col2.metric("Curtosis (Exceso)", f"{kurtosis(rendimientos):.2f}")
col3.metric("Sesgo", f"{skew(rendimientos):.2f}")

# ============================ PARTE (c): Cálculo VaR/ES =============================================
alphas = [0.95, 0.975, 0.99]
resultados = []
df_t = ajustar_t(rendimientos.values)

for alpha in alphas:
    # Histórico
    hVaR = rendimientos.quantile(1 - alpha)
    ES_hist = rendimientos[rendimientos <= hVaR].mean()
    
    # Normal
    VaR_norm = norm.ppf(1 - alpha, rendimientos.mean(), rendimientos.std())
    ES_norm = rendimientos[rendimientos <= VaR_norm].mean()
    
    # t-Student
    VaR_t = t.ppf(1 - alpha, df_t) * rendimientos.std() + rendimientos.mean()
    ES_t = rendimientos[rendimientos <= VaR_t].mean()
    
    # Monte Carlo (t-Student)
    sims = t.rvs(df_t, rendimientos.mean(), rendimientos.std(), size=10_000)
    VaR_mc = np.percentile(sims, (1 - alpha)*100)
    ES_mc = sims[sims <= VaR_mc].mean()
    
    resultados.append([alpha, hVaR, ES_hist, VaR_norm, ES_norm, VaR_t, ES_t, VaR_mc, ES_mc])

# Mostrar tabla
df_resultados = pd.DataFrame(
    resultados, 
    columns=["Alpha", "VaR Histórico", "ES Histórico", "VaR Normal", "ES Normal", 
             "VaR t-Student", "ES t-Student", "VaR Monte Carlo", "ES Monte Carlo"]
)
st.subheader("🔍 Comparación de Métodos de Cálculo")
st.dataframe(
    df_resultados.style.format("{:.2%}").background_gradient(cmap='Blues'),
    height=200
)

# ============================ PARTE (d): Rolling Windows ==========================================
window = 252
rolling_returns = rendimientos.rolling(window)

# Cálculos
rolling_VaRN = [
    rolling_returns.apply(lambda x: norm.ppf(1-a, x.mean(), x.std()))
    for a in [0.95, 0.99]
]

rolling_ESN = [
    rolling_returns.apply(lambda x: x[x <= norm.ppf(1-a, x.mean(), x.std())].mean())
    for a in [0.95, 0.99]
]

# Gráficos
st.subheader("📈 Rolling Windows: VaR y ES")
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
for i, a in enumerate([0.95, 0.99]):
    ax[i].plot(rendimientos.iloc[window:]*100, alpha=0.5, label='Retornos')
    ax[i].plot(rolling_VaRN[i].iloc[window:]*100, ls='--', label=f'VaR {a*100:.0f}%')
    ax[i].plot(rolling_ESN[i].iloc[window:]*100, ls=':', label=f'ES {a*100:.0f}%')
    ax[i].set_title(f'Nivel de confianza {a*100:.0f}%')
    ax[i].legend()
st.pyplot(fig)

# ============================ PARTE (f): VaR con Volatilidad Móvil ================================
st.subheader("📉 VaR con Volatilidad Móvil")
vol_movil = rendimientos.rolling(window=252).std()
var_vol_95 = norm.ppf(0.05) * vol_movil
var_vol_99 = norm.ppf(0.01) * vol_movil

# Gráfico
fig_vol, ax_vol = plt.subplots(figsize=(12, 6))
ax_vol.plot(rendimientos.iloc[252:]*100, alpha=0.5, label='Retornos')
ax_vol.plot(var_vol_95.iloc[252:]*100, label='VaR 95% (Vol Móvil)', ls='--')
ax_vol.plot(var_vol_99.iloc[252:]*100, label='VaR 99% (Vol Móvil)', ls=':')
ax_vol.set_title("VaR con Volatilidad Móvil (Distribución Normal)")
ax_vol.legend()
st.pyplot(fig_vol)

# ============================ PARTE (e): Violaciones ==============================================
st.subheader("🚨 Resultados de Violaciones")
violaciones = {
    'Normal 95%': (rendimientos.shift(-1) < rolling_VaRN[0]).sum(),
    'Normal 99%': (rendimientos.shift(-1) < rolling_VaRN[1]).sum(),
    'Vol Móvil 95%': (rendimientos.shift(-1) < var_vol_95).sum(),
    'Vol Móvil 99%': (rendimientos.shift(-1) < var_vol_99).sum()
}

for metodo, count in violaciones.items():
    porcentaje = (count / len(rendimientos)) * 100
    st.write(f"- **{metodo}:** {count} violaciones ({porcentaje:.2f}%)")