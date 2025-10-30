# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    df_raw = pd.read_csv("encuesta_musica.tab", delimiter='\\t', encoding='latin1')
    df = df_raw.copy()

    # --- Limpieza de columnas numéricas (géneros) ---
    genre_columns = [col for col in df.columns if col.startswith('MUSICA8_')]
    for col in genre_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Limpieza de columna de Edad y creación de grupos ---
    df['EDAD'] = pd.to_numeric(df['EDAD'], errors='coerce')
    df.dropna(subset=['EDAD'], inplace=True)
    df['EDAD'] = df['EDAD'].astype(int)
    
    df['GRUPOS_EDAD'] = pd.cut(df['EDAD'],
                              bins=[12, 17, 29, 49, 64, 100],
                              labels=['13-17', '18-29', '30-49', '50-64', '65+'],
                              right=True)

    # --- Limpieza de columnas de texto (categóricas) ---
    text_cols_to_clean = ['REGION', 'GENERO', 'MUSICA1', 'MUSICA9', 'MUSICA10']
    for col in text_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # --- Limpieza de columna de gasto (MUSICA13) ---
    if 'MUSICA13' in df.columns:
        df['MUSICA13'] = pd.to_numeric(df['MUSICA13'], errors='coerce').fillna(0)

    # --- Sumar las categorías de géneros ---
    for category, columns in GENRE_CATEGORIES.items():
        # Columna de suma (para conteos)
        df[category] = df[columns].sum(axis=1)
        # Columna binaria (para porcentajes/preferencia)
        df[f'CAT_{category}'] = (df[category] > 0).astype(int)

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

# =============================================================================
# DEFINICIÓN DE PESTAÑAS (TABS)
# =============================================================================
tabs_list = [
    "📈 Contador de Registros",
    "📊 Resumen General",
    "🧑‍🎤 Análisis por Edad",
    "🗺️ Análisis por Región",
    "🤘 Perfil Fan Rock",
    "📈 Tendencia Rock (Regresión)"
]
tabs = st.tabs(tabs_list)

# Asignar pestañas a variables
tab_contador = tabs[0]
tab_resumen = tabs[1]
tab_edad = tabs[2]
tab_region = tabs[3]
tab_perfil_rock = tabs[4]
tab_regresion_rock = tabs[5]

# =============================================================================
# PESTAÑA 1: CONTADOR
# =============================================================================
with tab_contador:
    st.header("Conteo Total de Registros")
    total_registros = len(df)
    st.metric(label="Total de Encuestas Válidas", value=f"{total_registros:,}")

# =============================================================================
# PESTAÑA 2: RESUMEN GENERAL
# =============================================================================
with tab_resumen:
    st.header("Resumen General del Consumo de Música")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Frecuencia de Escucha")
        
        # --- ¡CORRECCIÓN AQUÍ! ---
        # Usamos la columna 'MUSICA1' que sí existe y limpiamos
        if 'MUSICA1' in df.columns:
            frecuencia = df['MUSICA1'].value_counts()
            st.bar_chart(frecuencia)
        else:
            st.warning("Columna 'MUSICA1' (Frecuencia de escucha) no encontrada.")
        
    with col2:
        st.subheader("Géneros Más Populares (Total Oyentes)")
        popularidad_generos = df[genre_group_columns].sum().sort_values(ascending=False)
        data_to_plot = popularidad_generos.reset_index()
        data_to_plot.columns = ['Categoría Musical', 'Total de Oyentes']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Total de Oyentes', y='Categoría Musical', data=data_to_plot, palette='viridis', ax=ax)
        ax.set_title('Popularidad General de Géneros', fontsize=16)
        st.pyplot(fig)

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

# =============================================================================
# PESTAÑA 5: PERFIL FAN ROCK
# =============================================================================
with tab_perfil_rock:
    st.header("Análisis Específico - Perfil del Fan de Rock y Metal 🤘")

    # --- Crear la grilla 2x2 ---
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico 1: Preferencia por grupo de edad
        st.subheader("Preferencia por Grupo de Edad")
        rock_by_age = df.groupby('GRUPOS_EDAD', observed=True)['CAT_ROCK Y METAL'].mean() * 100
        rock_counts_by_age = df.groupby('GRUPOS_EDAD', observed=True)['CAT_ROCK Y METAL'].sum()
        
        age_order = ['13-17', '18-29', '30-49', '50-64', '65+']
        rock_by_age = rock_by_age.reindex(age_order).fillna(0)
        
        fig1, ax1 = plt.subplots()
        bars = ax1.bar(rock_by_age.index, rock_by_age.values,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F9D423'], alpha=0.8)
        ax1.set_title('PREFERENCIA DE ROCK Y METAL\nPOR GRUPO DE EDAD', fontweight='bold')
        ax1.set_ylabel('Porcentaje que Prefiere Rock (%)')
        ax1.grid(axis='y', alpha=0.3)
        for bar, percentage, age_group in zip(bars, rock_by_age.values, age_order):
            count = rock_counts_by_age.get(age_group, 0)
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{percentage:.1f}%\n({count} pers)', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig1)

    with col2:
        # Gráfico 2: Preferencia por Región
        st.subheader("Preferencia por Región")
        rock_by_region = df.groupby('REGION')['CAT_ROCK Y METAL'].mean() * 100
        rock_counts_by_region = df.groupby('REGION')['CAT_ROCK Y METAL'].sum()
        rock_by_region_sorted = rock_by_region.sort_values(ascending=False)
        
        fig2, ax2 = plt.subplots()
        bars = ax2.bar(rock_by_region_sorted.index, rock_by_region_sorted.values,
                       color=sns.color_palette("viridis", len(rock_by_region)), alpha=0.8)
        ax2.set_title('PREFERENCIA DE ROCK Y METAL\nPOR REGIÓN', fontweight='bold')
        ax2.set_ylabel('Porcentaje que Prefiere Rock (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        # Gráfico 3: Preferencia por Género Demográfico
        st.subheader("Preferencia por Género")
        if 'GENERO' in df.columns:
            rock_by_gender = df.groupby('GENERO')['CAT_ROCK Y METAL'].mean() * 100
            rock_counts_by_gender = df.groupby('GENERO')['CAT_ROCK Y METAL'].sum()
            
            fig3, ax3 = plt.subplots()
            colors = ['#FF9999', '#66B3FF', '#99FF99'] 
            bars = ax3.bar(rock_by_gender.index, rock_by_gender.values, 
                           color=colors[:len(rock_by_gender)], alpha=0.8)
            ax3.set_title('PREFERENCIA DE ROCK Y METAL\nPOR GÉNERO', fontweight='bold')
            ax3.set_ylabel('Porcentaje que Prefiere Rock (%)')
            ax3.grid(axis='y', alpha=0.3)
            st.pyplot(fig3)
        else:
            st.warning("Columna 'GENERO' no encontrada.")
        
    with col4:
        # Gráfico 4: Asistencia a Recitales
        st.subheader("Asistencia a Recitales (Fans vs. No Fans)")
        if 'MUSICA9' in df.columns:
            rock_fans = df[df['CAT_ROCK Y METAL'] == 1]
            non_rock_fans = df[df['CAT_ROCK Y METAL'] == 0]

            rock_attendance = rock_fans['MUSICA9'].value_counts(normalize=True) * 100
            non_rock_attendance = non_rock_fans['MUSICA9'].value_counts(normalize=True) * 100
            
            categories = ['Asisten a Recitales', 'No Asisten']
            rock_values = [rock_attendance.get('SI', 0), rock_attendance.get('NO', 0)]
            non_rock_values = [non_rock_attendance.get('SI', 0), non_rock_attendance.get('NO', 0)]
            
            fig4, ax4 = plt.subplots()
            x = np.arange(len(categories))
            width = 0.35
            bars1 = ax4.bar(x - width/2, rock_values, width, label='Fans Rock/Metal', color='#FF6B6B', alpha=0.8)
            bars2 = ax4.bar(x + width/2, non_rock_values, width, label='No Fans Rock/Metal', color='#4ECDC4', alpha=0.8)
            
            ax4.set_title('ASISTENCIA A RECITALES\nFANS vs NO FANS DE ROCK/METAL', fontweight='bold')
            ax4.set_ylabel('Porcentaje (%)')
            ax4.set_xticks(x, categories)
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            st.pyplot(fig4)
        else:
            st.warning("Columna 'MUSICA9' (Asistencia a recitales) no encontrada.")

    # --- Resumen Ejecutivo de la Pestaña ---
    st.markdown("---")
    st.header("Resumen Ejecutivo - Perfil del Fan de Rock y Metal")
    
    if 'GENERO' in df.columns:
        rock_fans = df[df['CAT_ROCK Y METAL'] == 1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Fans en Muestra", f"{len(rock_fans)}", f"{len(rock_fans)/len(df)*100:.1f}% del total")
        c2.metric("Grupo de Edad Más Afín", rock_by_age.idxmax(), f"{rock_by_age.max():.1f}%")
        c3.metric("Región Más Afin", rock_by_region.idxmax(), f"{rock_by_region.max():.1f}%")
        c4.metric("Género Más Afín", rock_by_gender.idxmax(), f"{rock_by_gender.max():.1f}%")
        
        st.subheader("Hallazgos Clave e Implicancias")
        st.info(f"""
        - **Perfil Principal:** El rock tiene su mayor penetración en el grupo de **{rock_by_age.idxmax()}**.
        - **Geografía:** La región **{rock_by_region.idxmax()}** lidera la preferencia por el rock en el país.
        - **Género:** Existe una brecha de preferencia de **{abs(rock_by_gender.diff().iloc[-1]):.1f}%** a favor de los **{rock_by_gender.idxmax()}**.
        - **Comportamiento:** Los fans del rock son significativamente **más propensos a asistir a recitales** que los no fans.
        
        **Implicancias Estratégicas:** Capitalizar la alta asistencia a recitales, fortalecer la presencia en la región líder ({rock_by_region.idxmax()}) 
        y crear estrategias para atraer a las audiencias con menor afinidad.
        """)
    else:
        st.warning("No se puede generar el resumen ejecutivo porque la columna 'GENERO' no existe.")

# =============================================================================
# PESTAÑA 6: TENDENCIA ROCK (REGRESIÓN)
# =============================================================================
with tab_regresion_rock: 
    st.header("Análisis de Tendencias - Rock & Metal 🎸")

    # 1. ANÁLISIS POR GRUPOS DE EDAD
    st.subheader("1. Preferencia por Grupo de Edad")
    rock_by_age = df.groupby('GRUPOS_EDAD', observed=True)['CAT_ROCK Y METAL'].mean() * 100
    st.dataframe(rock_by_age.map("{:.1f}%".format))

    # 2. PREPARANDO DATOS PARA REGRESIÓN
    st.subheader("2. Preparando Datos para Regresión")
    age_group_mapping = {
        '18-29': 23.5,
        '30-49': 39.5,
        '50-64': 57.0,
        '65+': 70.0
    }
    regression_data = []
    for age_group, age_value in age_group_mapping.items():
        if age_group in rock_by_age.index:
            preference = rock_by_age[age_group]
            regression_data.append({
                'edad_promedio': age_value,
                'preferencia_rock': preference,
                'grupo_edad': age_group
            })
    regression_df = pd.DataFrame(regression_data)
    st.dataframe(regression_df)

# 3. REGRESIÓN LINEAL
    st.subheader("3. Regresión Lineal - Preferencia vs. Edad")
    
    X = regression_df[['edad_promedio']]
    y = regression_df['preferencia_rock']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    intercept = model.intercept_
    slope = model.coef_[0]

    st.write("Resultados de la Regresión:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pendiente (Slope)", f"{slope:.4f}")
    col2.metric("R² (Precisión)", f"{r2:.4f}")
    col3.metric("Intercepto", f"{intercept:.4f}")
    
    st.write(f"**Ecuación:** Preferencia = {intercept:.2f} + {slope:.2f} * Edad")
    if slope < 0:
        st.warning(f"**Tendencia:** DECRECIENTE. Por cada año de edad, la preferencia disminuye en {abs(slope):.3f} puntos porcentuales.")
    else:
        st.success(f"**Tendencia:** CRECIENTE. Por cada año de edad, la preferencia aumenta en {slope:.3f} puntos porcentuales.")

    # 4. VISUALIZACIÓN DE LA TENDENCIA
    st.subheader("4. Visualización de la Tendencia")
    
    fig_reg, ax_reg = plt.subplots(figsize=(12, 8))
    
    # Gráfico de dispersión
    ax_reg.scatter(regression_df['edad_promedio'], regression_df['preferencia_rock'],
                   s=100, alpha=0.7, color='red', label='Datos observados')
    
    # Línea de regresión
    x_range = np.linspace(regression_df['edad_promedio'].min() - 5,
                          regression_df['edad_promedio'].max() + 10, 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax_reg.plot(x_range, y_range, 'b-', linewidth=2, label=f'Línea de Regresión (R² = {r2:.3f})')

    # Proyección futura
    edad_promedio_actual = df['EDAD'].mean()
    edad_5_anos = edad_promedio_actual + 5
    preferencia_5_anos = model.predict([[edad_5_anos]])[0]
    
    ax_reg.axvline(x=edad_5_anos, color='green', linestyle='--', alpha=0.7, label=f'Proyección a 5 años (Edad {edad_5_anos:.1f})')
    ax_reg.plot(edad_5_anos, preferencia_5_anos, 'go', markersize=10, label=f'Preferencia Proyectada: {preferencia_5_anos:.1f}%')

    ax_reg.set_xlabel('Edad Promedio del Grupo', fontsize=12)
    ax_reg.set_ylabel('Preferencia por Rock y Metal (%)', fontsize=12)
    ax_reg.set_title('TENDENCIA GENERACIONAL - ROCK Y METAL', fontsize=14, fontweight='bold')
    ax_reg.legend()
    ax_reg.grid(True, alpha=0.3)
    
    # Añadir etiquetas
    for i, row in regression_df.iterrows():
        ax_reg.annotate(row['grupo_edad'], (row['edad_promedio'], row['preferencia_rock']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    st.pyplot(fig_reg)

    # 5. RECOMENDACIONES ESTRATÉGICAS
    st.header("Recomendaciones Estratégicas")
    
    col1, col2, col3 = st.columns(3)
    rock_by_age_mean = rock_by_age.mean()
    rock_by_age_max_group = rock_by_age.idxmax()
    rock_by_age_max_val = rock_by_age.max()
    
    col1.metric("Preferencia Promedio", f"{rock_by_age_mean:.1f}%")
    col2.metric("Grupo Más Afín", rock_by_age_max_group, f"{rock_by_age_max_val:.1f}%")
    
    if slope < -0.1:
        st.warning("ALERTA: Fuerte tendencia decreciente generacional. Enfocarse en renovación de audiencia.")
    elif slope < -0.05:
        st.warning("TENDENCIA: Moderadamente decreciente. Balancear programación generacional.")
    else:
        st.success("TENDENCIA: Estable. Mantener y fortalecer la base de oyentes existente.")
        
    st.write("---")
    st.subheader("Acciones Recomendadas")
    st.info("""
    - Desarrollar contenidos que conecten con audiencias más jóvenes (ej: '18-29').
    - Crear puentes entre el rock clásico (atractivo para '50-64') y nuevos subgéneros.
    - Monitorear tendencias emergentes en subgéneros del rock para identificar oportunidades.
    """)