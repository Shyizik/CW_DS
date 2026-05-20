import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Налаштування веб-сторінки дашборду
st.set_page_config(page_title="Review Analysis", layout="wide")

# Заголовки інтерфейсу
st.title("Дашборд автоматизованої класифікації відгуків")
st.markdown("Система аналізу тональності відгуків на базі NLP моделі XLM-RoBERTa.")

# Ініціалізація та явне завантаження компонентів моделі в сесію
if 'classifier' not in st.session_state:
    with st.spinner("Завантаження NLP моделі..."):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        # Явне завантаження токенізатора та моделі без використання мета-тензорів
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            device_map=None
        )

        # Передача готових об'єктів у пайплайн
        st.session_state.classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )

classifier = st.session_state.classifier

# Мапінг вихідних міток моделі в українські категорії
label_mapping = {
    "positive": "Позитивний", "neutral": "Нейтральний", "negative": "Негативний",
    "LABEL_2": "Позитивний", "LABEL_1": "Нейтральний", "LABEL_0": "Негативний"
}

# Блок 1: Бокова панель для класифікації окремого відгуку
st.sidebar.header("Перевірити свій відгук")
user_input = st.sidebar.text_area("Введіть текст відгуку:")

if st.sidebar.button("Аналізувати"):
    if user_input.strip():
        res = classifier([user_input])[0]
        mapped_label = label_mapping.get(res['label'], res['label'])
        score = res['score']

        if mapped_label == "Позитивний":
            st.sidebar.success(f"Тональність: {mapped_label} (Впевненість: {score:.2%})")
        elif mapped_label == "Нейтральний":
            st.sidebar.warning(f"Тональність: {mapped_label} (Впевненість: {score:.2%})")
        else:
            st.sidebar.error(f"Тональність: {mapped_label} (Впевненість: {score:.2%})")
    else:
        st.sidebar.write("Поле вводу порожнє.")

# Блок 2: Формування базового датасету
data = {
    'review': [
        "Швидка доставка, товар якісний, повністю задоволений!",
        "Жахливий сервіс. Коробка прийшла пом'ята, а сам телефон не вмикається.",
        "Планшет непоганий за свої гроші, але екран міг би бути яскравішим.",
        "Навушники супер, звук чистий, рекомендую всем!",
        "Товар не відповідав опису, колір зовсім інший. Повернув назад.",
        "Звичайний чохол, нічого особливого. Свої функції виконує.",
        "Клавіатура зламалася на другий день використання. Якість жахлива.",
        "Все приїхало вчасно, дякую за оперативність!"
    ]
}
df = pd.DataFrame(data)

# Класифікація всього масиву даних
raw_results = classifier(df['review'].tolist())
df['predicted_label'] = [label_mapping.get(res['label'], res['label']) for res in raw_results]
df['confidence_score'] = [res['score'] for res in raw_results]

# Блок 3: Візуалізація метрик та загальної таблиці
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Розподіл тональності відгуків")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.countplot(
        x='predicted_label', data=df, ax=ax, palette='viridis',
        order=["Позитивний", "Нейтральний", "Негативний"], hue='predicted_label', legend=False
    )
    ax.set_xlabel("Категорія")
    ax.set_ylabel("Кількість відгуків")
    st.pyplot(fig)

with col2:
    st.subheader("Загальна база даних з мітками")
    st.dataframe(df, width=None)

# Блок 4: Розподіл відгуків за окремими категоріями
st.write("---")
st.subheader("Відгуки, відсортовані за категоріями")

tab_pos, tab_neu, tab_neg = st.tabs(["Позитивні", "Нейтральний", "Негативні"])

with tab_pos:
    pos_reviews = df[df['predicted_label'] == "Позитивний"]
    if not pos_reviews.empty:
        for idx, row in pos_reviews.iterrows():
            st.info(f"**Відгук:** {row['review']}  \n*Впевненість: {row['confidence_score']:.2%}*")
    else:
        st.write("Відгуки відсутні.")

with tab_neg:
    neg_reviews = df[df['predicted_label'] == "Негативний"]
    if not neg_reviews.empty:
        for idx, row in neg_reviews.iterrows():
            st.error(f"**Відгук:** {row['review']}  \n*Впевненість: {row['confidence_score']:.2%}*")
    else:
        st.write("Відгуки відсутні.")

with tab_neu:
    neu_reviews = df[df['predicted_label'] == "Нейтральний"]
    if not neu_reviews.empty:
        for idx, row in neu_reviews.iterrows():
            st.warning(f"**Відгук:** {row['review']}  \n*Впевненість: {row['confidence_score']:.2%}*")
    else:
        st.write("Відгуки відсутні.")