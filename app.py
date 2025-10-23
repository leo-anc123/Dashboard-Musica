# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Consumos Musicales en Argentina",
    page_icon="🎵",
    layout="wide"
)

# ==============================================================================
# CARGA Y LIMPIEZA DE DATOS (FUNCIÓN CACHEADA)
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    # Carga de datos
    df_raw = pd.read_csv("data/encuesta_musica.tab", delimiter='\t', encoding='latin1')
    df = df_raw.copy()

    # Identificar y limpiar columnas de géneros
    genre_columns = [col for col in df.columns if col.startswith('MUSICA8_')]
    for col in genre_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Creación de Grupos de Edad
    df['GRUPOS_EDAD'] = pd.cut(df['EDAD'],
                              bins=[12, 18, 30, 50, 65, 100],
                              labels=['13-17', '18-29', '30-49', '50-64', '65+'],
                              right=False)

    # Diccionario de Géneros y sus Agrupaciones
    genre_groups = {
        'FOLKLORE': ['MUSICA8_1', 'MUSICA8_2'],
        'ROCK': ['MUSICA8_3', 'MUSICA8_4'],
        'POP': ['MUSICA8_5'],
        'LATINA Y TROPICAL': ['MUSICA8_6', 'MUSICA8_10'],
        'ROMÁNTICA Y MELÓDICA': ['MUSICA8_7'],
        'JAZZ, BLUES Y SOUL': ['MUSICA8_8'],
        'URBANA (RAP/TRAP)': ['MUSICA8_9'],
        'ELECTRÓNICA': ['MUSICA8_11'],
        'TANGO': ['MUSICA8_12'],
        'CLÁSICA': ['MUSICA8_13']
    }

    # Crear las nuevas columnas de categorías sumando las originales
    for group_name, cols in genre_groups.items():
        df[group_name] = df[cols].sum(axis=1)

    return df

# Cargar los datos usando la función
df = load_and_clean_data()

# ==============================================================================
# CUERPO PRINCIPAL DE LA APLICACIÓN
# ==============================================================================
st.title("🎶 Dashboard de Consumos Musicales en Argentina")
st.markdown("Análisis interactivo de la **Encuesta Nacional de Consumos Culturales (2022-2023)**. Explora las tendencias y patrones de escucha a nivel nacional, por edad y por región.")

# Lista de Categorías de Género
genre_group_columns = [
    'FOLKLORE',
    'ROCK',
    'POP',
    'LATINA Y TROPICAL',
    'ROMÁNTICA Y MELÓDICA',
    'JAZZ, BLUES Y SOUL',
    'URBANA (RAP/TRAP)',
    'ELECTRÓNICA',
    'TANGO',
    'CLÁSICA'
]

# --- Pestañas para organizar el contenido ---
tab1, tab2, tab3 = st.tabs(["📊 Popularidad General", "🎂 Análisis por Edad", "🗺️ Análisis por Región"])

# ==============================================================================
# PESTAÑA 1: POPULARIDAD GENERAL
# ==============================================================================
with tab1:
    st.header("Ranking de Géneros Musicales a Nivel Nacional")
    group_popularity = df[genre_group_columns].sum().sort_values(ascending=False)

    fig_pop, ax_pop = plt.subplots(figsize=(12, 8))
    sns.barplot(x=group_popularity.values, y=group_popularity.index, ax=ax_pop, palette="viridis")
    ax_pop.set_title("Popularidad de Categorías Musicales en Argentina", fontsize=16)
    ax_pop.set_xlabel("Número Total de Oyentes (en la encuesta)", fontsize=12)
    for container in ax_pop.containers:
        ax_pop.bar_label(container, fmt='%d', padding=3)
    st.pyplot(fig_pop)

# ==============================================================================
# PESTAÑA 2: ANÁLISIS POR EDAD
# ==============================================================================
with tab2:
    st.header("Preferencia de Géneros Musicales por Grupo de Edad")
    genre_age = df.groupby('GRUPOS_EDAD', observed=True)[genre_group_columns].sum()
    genre_age_percent = genre_age.div(genre_age.sum(axis=1), axis=0) * 100

    st.subheader("Mapa de Calor: ¿Qué escucha cada generación?")
    fig_heat, ax_heat = plt.subplots(figsize=(14, 8))
    sns.heatmap(genre_age_percent.T, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=.5, ax=ax_heat)
    ax_heat.set_title("Preferencia Relativa de Géneros por Grupo de Edad (%)", fontsize=16)
    st.pyplot(fig_heat)
    
    st.subheader("Evolución de Preferencias a lo largo de la Vida")
    fig_line, ax_line = plt.subplots(figsize=(14, 8))
    genre_age_percent.plot(kind='line', marker='o', ax=ax_line)
    ax_line.set_title("Tendencia de Preferencia por Género según la Edad", fontsize=16)
    ax_line.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Categoría Musical', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_line)

# ==============================================================================
# PESTAÑA 3: ANÁLISIS POR REGIÓN
# ==============================================================================
with tab3:
    st.header("Distribución Geográfica de los Consumos Musicales")
    genre_region = df.groupby('REGION')[genre_group_columns].sum()
    genre_region_percent = genre_region.div(genre_region.sum(axis=1), axis=0) * 100

    st.subheader("Mapa de Calor: Preferencias por Región del País")
    fig_heat_reg, ax_heat_reg = plt.subplots(figsize=(14, 8))
    sns.heatmap(genre_region_percent.T, cmap="coolwarm", annot=True, fmt=".1f", linewidths=.5, ax=ax_heat_reg)
    ax_heat_reg.set_title("Preferencia Relativa de Géneros por Región (%)", fontsize=16)
    st.pyplot(fig_heat_reg)

# ==============================================================================
# SECCIÓN FINAL: EXPLORAR LOS DATOS
# ==============================================================================
with st.expander("Haz clic para ver una muestra de los datos limpios utilizados"):
    st.dataframe(df[genre_group_columns + ['GRUPOS_EDAD', 'REGION']].head(200))