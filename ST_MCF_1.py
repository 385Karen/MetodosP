import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm, t
import altair as alt



st.cache_data.clear()


st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

#######################################---BACKEND---##################################################


st.title("Visualización de Rendimientos de Acciones")
# st.write('hola')
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']


with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

    print(df_rendimientos)

#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Selecciona una acción de la lista ya que apartir de ella se calculara todo lo que se indica en cada ejercicio")

stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    skew = skew(df_rendimientos[stock_seleccionado])
    

    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{skew:.2}")

    #Calculo de Value-At-Risk y de Expected Shortfall (historico)

    std_dev = np.std(df_rendimientos[stock_seleccionado])

    # Definir niveles de confianza
    alphas = [0.95, 0.975, 0.99]
    resultados = []
    df_size = df_rendimientos[stock_seleccionado].size
    df_t = df_size - 1  # Grados de libertad para t-Student
    
    for alpha in alphas:
        # VaR y ES históricos
        hVaR = df_rendimientos[stock_seleccionado].quantile(1 - alpha)
        ES_hist = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR].mean()
        
        # VaR y ES paramétrico normal
        VaR_norm = norm.ppf(1 - alpha, rendimiento_medio, std_dev)
        ES_norm = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_norm].mean()
        
        # VaR y ES paramétrico t-Student
        t_ppf = t.ppf(1 - alpha, df_t)
        VaR_t = rendimiento_medio + std_dev * t_ppf * np.sqrt((df_t - 2) / df_t)
        ES_t = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_t].mean()
        
        # VaR y ES Monte Carlo
        simulaciones = np.random.normal(rendimiento_medio, std_dev, 10000)
        VaR_mc = np.percentile(simulaciones, (1 - alpha) * 100)
        ES_mc = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_mc].mean()
        
        resultados.append([alpha, hVaR, ES_hist, VaR_norm, ES_norm, VaR_t, ES_t, VaR_mc, ES_mc])
    
    # Crear DataFrame que contiene el VaR y ES para cada nivel de confianza

    df_resultados = pd.DataFrame(resultados, columns=["Alpha", "hVaR", "ES_hist", "VaR_Norm", "ES_Norm", "VaR_t", "ES_t", "VaR_MC", "ES_MC"])

    #Basicamente mostramos en patalla el dataframe antes creado

    st.subheader("Tabla comparativa de VaR y ES")
    st.text("Esta tabla muestra los resultados de los diferentes metodos de calculo de VaR y ES")

    #Mostramos el dataframe en pantalla de manera bonita
    st.dataframe(
    df_resultados.set_index("Alpha").style.format("{:.4%}")
    .applymap(lambda _: "background-color: #FFDDC1; color: black;", subset=["hVaR"])  # Durazno 
    .applymap(lambda _: "background-color: #C1E1FF; color: black;", subset=["ES_hist"])  # Azul 
    .applymap(lambda _: "background-color: #B5EAD7; color: black;", subset=["VaR_Norm"])  # Verde 
    .applymap(lambda _: "background-color: #FFB3BA; color: black;", subset=["ES_Norm"])  # Rosa 
    .applymap(lambda _: "background-color: #FFDAC1; color: black;", subset=["VaR_t"])  # Naranja 
    .applymap(lambda _: "background-color: #E2F0CB; color: black;", subset=["ES_t"])  # Verde 
    .applymap(lambda _: "background-color: #D4A5A5; color: black;", subset=["VaR_MC"])  # Rojo 
    .applymap(lambda _: "background-color: #CBAACB; color: black;", subset=["ES_MC"])  # Lila 
    )
    #Gráfico de barras para comparar los resultados
    st.subheader("Gráfico de comparación de VaR y ES")
    st.text("Este gráfico muestra la comparación de los diferentes métodos de cálculo de VaR y ES")
    st.bar_chart(df_resultados.set_index("Alpha").T)

    
    ##################################################################################################
    
    #Calculo de VaR y ES con Rolling Window

    st.subheader(f"Cálculo de VaR y ES con Rolling Window para {stock_seleccionado}")
    
    # Parámetros
    window_size = 252  # Ventana de 252 días
    alphas2 = [0.95, 0.99]

    VaR_rolling = {alpha: [] for alpha in alphas2}
    ES_rolling = {alpha: [] for alpha in alphas2}
    fechas = df_rendimientos.index[window_size:]
    
    
    # Cálculo de VaR y ES con rolling window
    for i in range(len(fechas)):
        ventana = df_rendimientos.iloc[i:i+window_size]
        Media_R = ventana.mean()
        Std_R = ventana.std()
        
        for alpha in alphas2:
            var_param = norm.ppf(1 - alpha, Media_R, Std_R)
            es_param = ventana[ventana <= var_param].mean()
            
            VaR_rolling[alpha].append(var_param)
            ES_rolling[alpha].append(es_param)



    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fechas, df_rendimientos.iloc[window_size:], label='Retornos', color='blue', alpha=0.5)
    ax.plot(fechas, VaR_rolling[0.95], label='VaR 95%', color='red')
    ax.plot(fechas, ES_rolling[0.95], label='ES 95%', color='darkred', linestyle='dashed')
    ax.plot(fechas, VaR_rolling[0.99], label='VaR 99%', color='green')
    ax.plot(fechas, ES_rolling[0.99], label='ES 99%', color='darkgreen', linestyle='dashed')
    
    ax.set_title(f'VaR y ES con Rolling Window ({stock_seleccionado})')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Retornos')
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)