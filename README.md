# Дашборд для прогнозирования выработки электроэнергии

Веб-приложение на базе Streamlit для инференса моделей машинного обучения и экспресс-анализа технологических параметров электростанции. 

## О проекте
Цель работы - построение и деплой моделей регрессии, предсказывающих объем выработки энергии (целевой признак: `Appliances`) на основе показателей климатических датчиков, внутренней среды помещений и временных меток.

### Стек технологий:
* **Frontend/App framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **ML Models:** XGBoost, Scikit-Learn (Trees, Ensembles, MLP)
* **Visualization:** Matplotlib, Seaborn

## Структура репозитория
- `app.py` - основной файл веб-приложения Streamlit.
- `dataR.csv` - датасет с техническими и погодными метриками.
- `scaler.pkl` - сериализованный StandardScaler для подготовки признаков.
- `ML3_XGBoost.json` - сохраненные веса модели XGBoost.
- `ML1_DecisionTree.pkl`, `ML2_GradientBoosting.pkl` и др. - файлы предобученных моделей.
- `requirements.txt` - список зависимостей для сборки окружения.
- `photo.jpg` - фото автора.
  
## Локальный запуск

1. Склонируйте репозиторий:
```bash
git clone https://github.com/nameofmypony/energy-prediction-rgr
cd energy-prediction-rgr
```

2. Установите необходимые пакеты:
```bash
pip install -r requirements.txt
```

3. Запустите локальный сервер Streamlit:
```bash
streamlit run app.py
```
