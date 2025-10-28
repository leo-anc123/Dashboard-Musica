import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard de Consumos Musicales en Argentina",
    page_icon="🎵",
    layout="wide"
)

# =============================================================================
# DEFINICIÓN DE CATEGORÍAS
# =============================================================================
GENRE_CATEGORIES = {
    'MÚSICA URBANA': ['MUSICA8_1', 'MUSICA8_2', 'MUSICA8_3', 'MUSICA8_13'],
    'ROCK Y METAL': ['MUSICA8_4', 'MUSICA8_5', 'MUSICA8_6'],
    'FOLKLORE Y TANGO': ['MUSICA8_7', 'MUSICA8_8'],
    'POP Y ELECTRÓNICA': ['MUSICA8_9', 'MUSICA8_10'],
    'CUMBIA': ['MUSICA8_11', 'MUSICA8_12'],
}

# =============================================================================
# CARGA Y LIMPIEZA DE DATOS (FUNCIÓN CACHEADA)
# =============================================================================
@st.cache_data
def load_and_clean_data():
    # Carga de datos
    df_raw = pd.read_csv("encuesta_musica.tab", delimiter='\\t', encoding='latin1')
    df = df_raw.copy()

    # Identificar y limpiar columnas de géneros
    genre_columns = [col for col in df.columns if col.startswith('MUSICA8_')]
    for col in genre_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Creación de Grupos de Edad
    df['EDAD'] = pd.to_numeric(df['EDAD'], errors='coerce')
    df.dropna(subset=['EDAD'], inplace=True)
    df['EDAD'] = df['EDAD'].astype(int)
    df['GRUPOS_EDAD'] = pd.cut(df['EDAD'],
                              bins=[12, 17, 29, 49, 64, 100],
                              labels=['13-17', '18-29', '30-49', '50-64', '65+'],
                              right=True)
    df['REGION'] = df['REGION'].str.strip()

    # Sumar las categorías usando el diccionario global
    for category, columns in GENRE_CATEGORIES.items():
        df[category] = df[columns].sum(axis=1)

    return df, list(GENRE_CATEGORIES.keys())

# --- Carga de datos principal ---
df, genre_group_columns = load_and_clean_data()

# =============================================================================
# TÍTULO Y DESCRIPCIÓN
# =============================================================================
st.title("Dashboard de Consumos Musicales en Argentina 🎵")
st.write("""
Análisis de los datos de la Encuesta Nacional de Consumos Culturales (ENCC) 2022-23.
...
""")
st.subheader("¿El rock ha muerto?")

# =============================================================================
# DEFINICIÓN DE PESTAÑAS (TABS)
# =============================================================================
tab_contador, tab_resumen, tab_edad, tab_region = st.tabs([
    "📈 Contador de Registros",
    "📊 Resumen General",
    "🧑‍🎤 Análisis por Edad",
    "🗺️ Análisis por Región",
])

# =============================================================================
# PESTAÑA 1: CONTADOR DE REGISTROS
# =============================================================================
with tab_contador:
    st.header("Conteo Total de Registros")
    total_registros = len(df)
    st.metric(label="Total de Encuestas Válidas", value=f"{total_registros:,}")
    st.info("Este número representa el total de filas (participantes) en el conjunto de datos...")

# =============================================================================
# PESTAÑA 2: RESUMEN GENERAL (¡MODIFICACIÓN AQUÍ!)
# =============================================================================
with tab_resumen:
    st.header("Resumen General del Consumo de Música")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frecuencia de Escucha")
        frecuencia = df['MUSICA1'].value_counts()
        st.bar_chart(frecuencia)
        
    with col2:
        # --- GRÁFICO ESTILIZADO DE SEABORN ---
        st.subheader("Géneros Más Populares (Total Oyentes)")
        
        # 1. Preparamos los datos
        popularidad_generos = df[genre_group_columns].sum().sort_values(ascending=False)
        # Convertimos la Serie de Pandas a un DataFrame para Seaborn
        data_to_plot = popularidad_generos.reset_index()
        data_to_plot.columns = ['Categoría Musical', 'Total de Oyentes']

        # 2. Creamos la figura y el gráfico
        fig, ax = plt.subplots(figsize=(10, 6)) # Ajusta el tamaño como quieras
        sns.barplot(
            x='Total de Oyentes', 
            y='Categoría Musical', 
            data=data_to_plot, 
            palette='viridis', # Puedes cambiar 'viridis' por 'plasma', 'coolwarm', etc.
            ax=ax
        )
        
        # 3. Estilizamos el gráfico
        ax.set_title('Popularidad General de Géneros', fontsize=16)
        ax.set_xlabel('Total de Oyentes', fontsize=12)
        ax.set_ylabel('Categoría', fontsize=12)
        sns.despine(left=True, bottom=True) # Quita los bordes feos
        
        # 4. Mostramos el gráfico en Streamlit
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Datos Completos (Filtrados)")
    st.dataframe(df)

# =============================================================================
# PESTAÑA 3: ANÁLISIS POR EDAD
# =============================================================================
with tab_edad:
    st.header("Exploración de Preferencias por Grupo de Edad")
    genre_age = df.groupby('GRUPOS_EDAD', observed=True)[genre_group_columns].sum()
    
    st.subheader("Mapa de Calor: Intensidad de Preferencia (Edad vs. Género)")
    fig_heat_age, ax_heat_age = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_age.T, cmap="viridis", annot=True, fmt=".0f", linewidths=.5, ax=ax_heat_age)
    st.pyplot(fig_heat_age)
    
    st.markdown("---")
    st.subheader("Tendencia de Géneros por Edad")
    fig_line, ax_line = plt.subplots(figsize=(12, 6))
    genre_age.plot(kind='line', marker='o', ax=ax_line)
    plt.legend(title='Categoría Musical', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_line)

# =============================================================================
# PESTAÑA 4: ANÁLISIS POR REGIÓN
# =============================================================================
with tab_region:
    st.header("Distribución Geográfica de los Consumos Musicales")
    genre_region = df.groupby('REGION')[genre_group_columns].sum()
    genre_region_percent = genre_region.div(genre_region.sum(axis=1), axis=0) * 100
    
    st.subheader("Mapa de Calor: Preferencias por Región del País")
    fig_heat_reg, ax_heat_reg = plt.subplots(figsize=(14, 8))
    sns.heatmap(genre_region_percent.T, cmap="coolwarm", annot=True, fmt=".1f", linewidths=.5, ax=ax_heat_reg)
    st.pyplot(fig_heat_reg)

