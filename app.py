import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

@st.cache_data
def load_data():
    df = pd.read_csv('dataR.csv', sep=",", encoding="utf-8")
    return df

st.set_page_config(
    page_title="РГР",
    layout="wide"
)

page = st.sidebar.radio(
    "Навигация",
    ["Главная", "Датасет", "Визуализация", "Предсказание"]
)

if page == "Главная":
    st.title("Расчетно-графическая работа")
    st.subheader("по дисциплине: Машинное обучение и большие данные")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Информация об авторе")
        st.write("Студент: Безверхний Роман Сергеевич")
        st.write("Курс: 2")
        st.write("Группа: ФИТ-241")
        
        st.markdown("### Тема работы")
        st.write(
            "Разработка веб-приложения для инференса моделей машинного обучения и анализа данных."
        )
        st.markdown("### Задача")
        st.write(
            "На основе показателей датчиков температуры, влажности и погодных условий построить "
            "модели регрессии для предсказания выработка энергии электростанцией (целевой признак: `Appliances`)."
        )
        
    with col2:
        st.image("photo.jpg", caption="Безверхний Р. С.", width=200)

elif page == "Датасет":
    st.title("Описание набора данных")
    st.write("В данном разделе представлена информация о структуре датасета о выработке энергии электростанцией.")
    
    st.subheader("Словарь признаков")
    st.markdown("""
    - `Appliances` - выработка электроэнергии станцией в кВт·ч (Целевая переменная)
    - `lights` - потребление энергии осветительными приборами в кВт·ч
    - `T1` - `T9` - показания датчиков температуры в различных зонах (°C)
    - `RH_1` - `RH_9` - показания датчиков влажности в различных зонах (%)
    - `T_out` - температура на улице (°C)
    - `Press_mm_hg` - атмосферное давление (мм рт. ст.)
    - `RH_out` - влажность на улице (%)
    - `Windspeed` - скорость ветра (м/с)
    - `Visibility` - видимость (км)
    - `Tdewpoint` - точка росы (°C)
    - `month`, `day`, `hour`, `minute` - временные параметры
    """)
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Количество записей: {df.shape[0]}")
    with col2:
        st.info(f"Количество признаков: {df.shape[1]}")
        
    st.subheader("Датасет (первые 100 строк)")
    st.dataframe(df.head(100))
    
    st.subheader("Статистическое описание признаков")
    st.dataframe(df.describe())
    
    st.subheader("Распределение целевого признака `Appliances`")
    st.write("Гистограмма распределения выработки электроэнергии:")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Appliances'], bins=50, kde=True, ax=ax, color='teal')
    ax.set_xlabel("Энергия (кВт·ч)")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

elif page == "Визуализация":
    df = load_data()
    st.title("Статистический анализ и визуализация")
    st.write("Графики зависимостей и распределений признаков в наборе данных")
    
    st.subheader("1. Матрица корреляции ключевых признаков")
    cols_for_corr = ['Appliances', 'lights', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed']
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols_for_corr].corr(), annot=True, cmap='seismic', center=0, ax=ax1, fmt=".2f")
    st.pyplot(fig1)
    
    st.subheader("2. Зависимость выработки электроэнергии от часа суток (`Hour`)")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='hour', y='Appliances', data=df, ax=ax2)
    ax2.set_xlabel("Час суток")
    ax2.set_ylabel("Выработка электроэнергии (кВт·ч)")
    st.pyplot(fig2)
    
    st.subheader("3. Взаимосвязь уличной температуры и выработки электроэнергии")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='T_out', y='Appliances', data=df, alpha=0.3, ax=ax3)
    ax3.set_xlabel("Температура на улице (°C)")
    ax3.set_ylabel("Выработка электроэнергии (кВт·ч)")
    st.pyplot(fig3)

    st.subheader("4. Зависимость выработки электроэнергии от влажности снаружи (`RH_out`)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    df_smoothed = df.copy()
    df_smoothed['RH_out'] = df_smoothed['RH_out'].round()
    sns.lineplot(x='RH_out', y='Appliances', data=df_smoothed, ax=ax4)
    ax4.set_xlabel("Влажность на улице (%)")
    ax4.set_ylabel("Выработка электроэнергии (кВт·ч)")
    st.pyplot(fig4)

elif page == "Предсказание":
    st.title("Предсказание выработки электроэнергии станцией") 
    
    model_choice = st.selectbox(
        "Выберите модель машинного обучения:",
        ["ML1_DecisionTree", "ML2_GradientBoosting", "ML3_XGBoost", "ML4_Bagging", "ML5_Stacking", "ML6_NeuralNetwork"]
    )
    
    if model_choice == "ML3_XGBoost":
        model = xgb.XGBRegressor()
        model.load_model('ML3_XGBoost.json')
    else: 
        model = joblib.load(f"{model_choice}.pkl")
    scaler = joblib.load("scaler.pkl")

    df_raw = load_data()
    X_raw = df_raw.drop(columns=['Appliances'])
    expected_columns = X_raw.columns
    
    tab1, tab2 = st.tabs(["Ввод данных вручную", "Загрузка файла (*.csv)"])
    
    with tab1:
        st.subheader("Показания датчиков")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            lights = st.slider("Lights (кВт·ч)", 0.0, 100.0, 30.0)
            T1 = st.slider("T1 (°C)", 10.0, 30.0, 19.89)
            RH_1 = st.slider("RH_1 (%)", 20.0, 90.0, 47.59)
            T2 = st.slider("T2 (°C)", 10.0, 30.0, 19.2)
            RH_2 = st.slider("RH_2 (%)", 20.0, 90.0, 44.79)
            T3 = st.slider("T3 (°C)", 10.0, 30.0, 19.79)
            RH_3 = st.slider("RH_3 (%)", 20.0, 90.0, 44.73)
            
        with col2:
            T4 = st.slider("T4 (°C)", 10.0, 30.0, 19.0)
            RH_4 = st.slider("RH_4 (%)", 20.0, 90.0, 45.56)
            T5 = st.slider("T5 (°C)", 10.0, 30.0, 17.16)
            RH_5 = st.slider("RH_5 (%)", 20.0, 90.0, 55.2)
            T6 = st.slider("T6 (°C)", -5.0, 30.0, 7.02)
            RH_6 = st.slider("RH_6 (%)", 0.0, 100.0, 84.25)
            T7 = st.slider("T7 (°C)", 10.0, 30.0, 17.2)
            
        with col3:
            RH_7 = st.slider("RH_7 (%)", 20.0, 90.0, 41.62)
            T8 = st.slider("T8 (°C)", 10.0, 30.0, 18.2)
            RH_8 = st.slider("RH_8 (%)", 20.0, 90.0, 48.9)
            T9 = st.slider("T9 (°C)", 10.0, 30.0, 17.03)
            RH_9 = st.slider("RH_9 (%)", 20.0, 90.0, 45.53)
            T_out = st.slider("T_out (°C)", -10.0, 30.0, 6.6)
            Press_mm_hg = st.slider("Давление (мм)", 700.0, 800.0, 733.5)
            
        with col4:
            RH_out = st.slider("RH_out (%)", 0.0, 100.0, 92.0)
            Windspeed = st.slider("Ветер (м/с)", 0.0, 20.0, 7.0)
            Visibility = st.slider("Видимость (км)", 0.0, 70.0, 63.0) 
            Tdewpoint = st.slider("Точка росы", -10.0, 20.0, 5.3)
            month = st.slider("Месяц", 1, 12, 1)
            day = st.slider("День", 1, 31, 11)
            hour = st.slider("Час", 0, 23, 17)
            minute = st.slider("Минута", 0, 59, 0)

        if st.button("Сделать предсказание", type="primary"):
            input_data = pd.DataFrame({
                'lights': [lights], 'T1': [T1], 'RH_1': [RH_1], 'T2': [T2], 'RH_2': [RH_2],
                'T3': [T3], 'RH_3': [RH_3], 'T4': [T4], 'RH_4': [RH_4], 'T5': [T5],
                'RH_5': [RH_5], 'T6': [T6], 'RH_6': [RH_6], 'T7': [T7], 'RH_7': [RH_7],
                'T8': [T8], 'RH_8': [RH_8], 'T9': [T9], 'RH_9': [RH_9], 'T_out': [T_out],
                'Press_mm_hg': [Press_mm_hg], 'RH_out': [RH_out], 'Windspeed': [Windspeed],
                'Visibility': [Visibility], 'Tdewpoint': [Tdewpoint], 'month': [month],
                'day': [day], 'hour': [hour], 'minute': [minute]
            })
            
            input_aligned = input_data.reindex(columns=expected_columns, fill_value=0)
            if model_choice == "ML6_NeuralNetwork":
                input_scaled = scaler.transform(input_aligned)
                input_df = pd.DataFrame(input_scaled, columns=expected_columns)
                prediction = model.predict(input_df.values)[0]
            else:
                input_df = pd.DataFrame(input_aligned, columns=expected_columns)
                prediction = model.predict(input_df)[0]

            st.success(f"Ожидаемая выработка электроэнергии: {prediction:.2f} кВт·ч")

    with tab2:
        st.info("Формат файла: CSV с теми же колонками (кроме Appliances), что и в оригинальном датасете")
        uploaded_file = st.file_uploader("Загрузите файл *.csv", type="csv")
        
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            st.write("Превью загруженных данных:")
            st.dataframe(user_df.head())
            
            if st.button("Предсказать для загруженного файла"):
                if 'Appliances' in user_df.columns:
                    user_df = user_df.drop(columns=['Appliances'])

                user_aligned = user_df.reindex(columns=expected_columns, fill_value=0)
                if model_choice == "ML6_NeuralNetwork":
                    user_scaled = scaler.transform(user_aligned)
                    user_df = pd.DataFrame(user_scaled, columns=expected_columns)
                    predictions = model.predict(user_df.values)
                else:
                    user_df = pd.DataFrame(user_aligned, columns=expected_columns)
                    predictions = model.predict(user_df)
                
                user_df['Предсказание Appliances'] = [f"{p:.2f} кВт·ч" for p in predictions]
                st.success("Прогнозирование успешно завершено")
                st.dataframe(user_df)