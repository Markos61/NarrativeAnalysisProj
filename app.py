import numpy as np
import streamlit as st
from file_utils import extract_text
from similarity_analisys_funcs import similarity_start
from classifier_funcs import get_prediction
from similarity_funcs import get_example_narratives

# streamlit run app.py

st.set_page_config(page_title="Narrative Analysis Tools")  # layout="wide"

uploaded_file = st.file_uploader(
    "Загрузите файл (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"]
)

if uploaded_file:
    # если новый файл, сохраняем текст в session_state
    if st.session_state.get("filename") != uploaded_file.name:
        st.session_state.filename = uploaded_file.name
        st.session_state.text = extract_text(uploaded_file)

# если текст ещё не был загружен, задаём пустую строку
text = st.session_state.get("text", "")
file_key = uploaded_file.name if uploaded_file else "empty"

# ----------------- Общий блок с текстом -----------------
if not text:
    st.info("Загрузите файл")
else:
    st.text_area(
        "Предпросмотр загруженного текста",
        value=text,
        height=250,
        key=f"text_tab1_{file_key}",  # динамический ключ
        disabled=False  # если нужно только для просмотра
    )

st.divider()

# ----------------- Вкладки -----------------
tab1, tab2 = st.tabs(["Идеологическая окраска", "Анализ сходства"])

CLASS_NAMES = [
    "Неолиберализм",
    "Социализм",
    "Дирижизм",
    "Особый путь",
    "Экологизм",
    "Не определена"
]

CLASS_COLORS = {
    "Неолиберализм": "#e5c500",
    "Социализм": "#f91f27",
    "Дирижизм": "#f69c22",
    "Особый путь": "#1154aa",
    "Экологизм": "#2bb506",
    "Не определена": "#585859"
}

# ----------------- Tab 1 -----------------
with tab1:
    if not text:
        st.info("Загрузите файл")
    else:
        models_names = ["LSTM на статьях", "LSTM на стенограммах"]
        model_name = st.selectbox("Выберите классификатор:", models_names)

        if model_name == "LSTM на стенограммах":
            model_path = 'LSTM_gos_duma.h5'
        else:
            model_path = 'LSTM_articles.h5'

        st.divider()
        classification_button = st.button("Начать классификацию")

        if classification_button:
            prediction = get_prediction(model_path, text)

            pred = np.array(prediction)
            class_id = pred.argmax()
            confidence = pred[0][class_id] * 100
            if confidence >= 50:
                class_name = CLASS_NAMES[class_id]
                color = CLASS_COLORS[class_name]

                # Анимированная карточка
                st.markdown(
                    f"""
                    <style>
                    .result-card {{
                        animation: fadeInUp 0.6s ease-out;
                    }}

                    @keyframes fadeInUp {{
                        from {{
                            opacity: 0;
                            transform: translateY(20px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                    }}
                    </style>

                    <div class="result-card" style="
                        padding:14px;
                        border-left:6px solid {color};
                        background:#fbfbfb;
                        border-radius:8px;
                        margin-top:10px;
                        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                    ">
                        <div style="font-size:19px; font-weight:bold; color:{color};">
                                        Идеология документа — {class_name} ({confidence:.1f}%)
                                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                class_name = CLASS_NAMES[len(CLASS_NAMES) - 1]
                color = CLASS_COLORS[class_name]

                st.markdown(
                    f"""
                    <style>
                    .result-card {{
                        animation: fadeInUp 0.6s ease-out;
                    }}

                    @keyframes fadeInUp {{
                        from {{
                            opacity: 0;
                            transform: translateY(20px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                    }}
                    </style>

                    <div class="result-card" style="
                        padding:14px;
                        border-left:6px solid {color};
                        background:#fbfbfb;
                        border-radius:8px;
                        margin-top:10px;
                        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                    ">
                        <div style="font-size:19px; font-weight:bold; color:{color};">
                                        Идеология документа — {class_name}
                                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ----------------- Tab 2 -----------------
# ---------- Инициализация ----------
if "narratives" not in st.session_state:
    st.session_state.narratives = []

with tab2:
    if not text:
        st.info("Загрузите файл")
    else:
        #
        predefined_phrases = get_example_narratives()
        # Объединяем с опцией для ручного ввода
        options = ["Ввести свой текст"] + predefined_phrases

        # Выбор из списка
        selected_option = st.selectbox("Выберите фразу или 'Ввести свой текст':", options, key="select_phrase")

        # Текстовое поле для ручного ввода
        new_phrase = ""
        if selected_option == "Ввести свой текст":
            new_phrase = st.text_input(
                "Введите фразу (субъект, действие, объект)",
                key="phrase_input"
            )
        else:
            new_phrase = selected_option
        #

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Добавить фразу", key="add_similarity"):
                if new_phrase.strip() and new_phrase not in st.session_state.narratives and len(st.session_state.narratives) < 11:
                    st.session_state.narratives.append(new_phrase)

        with col2:
            if st.button("Очистить список", key="clear_similarity"):
                st.session_state.narratives.clear()

        # ---------- Отображение списка ----------
        if st.session_state.narratives:
            st.markdown("### Выбранные для анализа фразы:")
            for i, phrase in enumerate(st.session_state.narratives, 1):
                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        border-radius:8px;
                        background:#fbfbfb;
                        margin-bottom:6px;
                    ">
                        <b>{i}.</b> {phrase}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()

        # ---------- Запуск анализа ----------
        if st.button("Начать анализ сходства", key="run_similarity"):
            if not st.session_state.narratives:
                st.warning("Добавьте хотя бы одну фразу.")
            else:
                narratives = []
                for nar in st.session_state.narratives:
                    narratives.append(str(nar))
                result = similarity_start([text], narratives)
                st.divider()

                for idx, df in enumerate(result, start=0):
                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            border-left:6px solid #1357a5;
                            background:#fbfbfb;
                            border-radius:8px;
                            color:#131313;
                            font-weight:bold;
                            font-size:18px;
                            margin-bottom:10px;
                            text-align: center;
                        ">
                            <div style="font-size:14px; font-weight:bold; color:#161616;">
                                            Наиболее близкие предложения к
                            </div>
                            <div style="margin-top:6px; font-size:17px; color:#1357a5;">
                                            {narratives[idx]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Выбираем только нужные колонки
                    if "sentence" in df.columns and "connected sentences" in df.columns:
                        if df[["sentence", "connected sentences"]].notna().any().any():
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.write('Сходство не найдено')

                    try:
                        if df[["sentence", "connected sentences"]].notna().any().any():
                            st.markdown(
                                f"""
                                                    <div style="
                                                        padding:12px;
                                                        border-left:6px solid #1357a5;
                                                        background:#fbfbfb;
                                                        border-radius:8px;
                                                        color:#131313;
                                                        font-weight:bold;
                                                        font-size:18px;
                                                        margin-bottom:10px;
                                                        text-align: center;
                                                    ">
                                                        ТОП-3 предложения
                                                    </div>
                                                    """,
                                unsafe_allow_html=True
                            )
                            count = 0
                            for _, row in df.iterrows():
                                count += 1
                                if count == 4:
                                    break
                                sentence = row.get("sentence", "")
                                connected = row.get("connected sentences", "")

                                st.markdown(
                                    f"""
                                    <div style="
                                        padding:12px;
                                        background:#fbfbfb;
                                        border-radius:8px;
                                        color:#131313;
                                        margin-bottom:10px;
                                        text-align:left;
                                    ">
                                        <div style="font-size:14px; font-weight:bold; color:#161616;">
                                            {sentence.replace('<', '&lt;').replace('>', '&gt;')}
                                        </div>
                                        <div style="margin-top:6px; font-size:13px; color:#212121;">
                                            {connected.replace('<', '&lt;').replace('>', '&gt;')}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    except:
                        pass

                    st.divider()
