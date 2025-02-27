########################
# word2earn_app.py
########################

import streamlit as st
import datetime
import sqlite3
import unicodedata
import io
import random
import time

# NLP и перевод
import spacy
from googletrans import Translator
from gtts import gTTS

# Grammar check
import language_tool_python

# synonyms from TextBlob/WordNet
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
from textblob import Word, TextBlob

# Data manipulations
import pandas as pd
import numpy as np

# ollama for local LLM
import ollama

# ---------------------- #
#   Инициализация
# ---------------------- #
st.set_page_config(
    page_title="KON",
    page_icon="💰",
    layout="wide"
)

# Инициализируем spacy-модель
nlp = spacy.load("en_core_web_sm")

# Инициализируем переводчик и LanguageTool
translator = Translator()
lang_tool = language_tool_python.LanguageTool("en-US")

# Подключаемся к SQLite
conn = sqlite3.connect("word2earn_prod.db", check_same_thread=False)
c = conn.cursor()

# Создаём таблицы (если не существуют)
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    settings TEXT,
    level TEXT,
    tokens REAL DEFAULT 0,
    date_joined TIMESTAMP
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    word TEXT,
    translation TEXT,
    date_added TIMESTAMP,
    times_correct INTEGER DEFAULT 0,
    times_incorrect INTEGER DEFAULT 0
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    role TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    forwarded BOOLEAN DEFAULT 0,
    audio BOOLEAN DEFAULT 0
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS quizzes (
    quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    date_taken TIMESTAMP,
    score INTEGER
)
""")
conn.commit()


# ---------------------- #
#  Вспомогательные функции
# ---------------------- #

def get_current_user_id() -> int:
    """
    Упрощённо: считаем, что пользователь уже авторизован и user_id=1.
    В реальном продакшене - делать полноценную аутентификацию.
    """
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = 1
        c.execute("SELECT * FROM users WHERE user_id=?", (1,))
        if not c.fetchone():
            c.execute("""
            INSERT INTO users (username, password, settings, level, tokens, date_joined)
            VALUES (?,?,?,?,?,?)
            """, ("demo_user", "demo_pass", "{}", "A1", 0, datetime.datetime.now()))
            conn.commit()
    return st.session_state["user_id"]


def normalize_text(text: str) -> str:
    """Убираем диакритику, приводим к ASCII."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()


def translate_text(text: str, dest="ru") -> str:
    """Перевод произвольного текста через googletrans."""
    try:
        translation = translator.translate(text, dest=dest)
        return translation.text
    except Exception:
        return "[Error translating text]"


def generate_tts(text: str, lang="en"):
    """Генерация озвучки через gTTS."""
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


def correct_text_with_languagetool(text: str) -> str:
    """
    Используем language_tool_python для нахождения ошибок.
    Возвращаем текст с зачёркиваниями и исправлениями (Markdown).
    """
    matches = lang_tool.check(text)
    if not matches:
        return text

    corrected_text = text
    offset = 0
    for match in reversed(matches):
        start = match.offset
        end = match.offset + match.errorLength
        original = corrected_text[start + offset: end + offset]
        replacement = match.replacements[0] if match.replacements else original
        new_fragment = f"~~{original}~~ **{replacement}**"
        corrected_text = (corrected_text[:start + offset]
                          + new_fragment
                          + corrected_text[end + offset:])
        offset += len(new_fragment) - (end - start)
    return corrected_text


def categorize_message(text: str) -> str:
    """Простая категоризация."""
    lower = text.lower()
    if any(w in lower for w in ["travel", "vacation", "flight", "museum"]):
        return "Travel"
    elif any(w in lower for w in ["food", "eat", "restaurant", "meal"]):
        return "Food"
    elif any(w in lower for w in ["sport", "game", "football", "basketball"]):
        return "Sports"
    elif any(w in lower for w in ["work", "job", "office", "project"]):
        return "Work"
    elif any(w in lower for w in ["study", "university", "exam", "english"]):
        return "Education"
    else:
        return "General"


def fuzzy_normalize_speech(recognized_text: str) -> str:
    """Заглушка для нормализации произношения."""
    return recognized_text


def get_synonyms_en(word: str) -> list:
    """Находим синонимы через TextBlob/WordNet."""
    w = Word(word)
    synsets = w.synsets
    if synsets:
        lemmas = synsets[0].lemmas()
        synonyms = [lemma.name().replace("_", " ") for lemma in lemmas]
        return list(set(synonyms))
    return []


def get_idioms_mock(word: str) -> list:
    """Заглушка идиом."""
    return [f"{word} away", f"pull a {word}", f"hit the {word}"]


def get_examples_mock(word: str) -> list:
    """Заглушка примеров."""
    return [
        f"I really like the word '{word}'!",
        f"{word} can improve your speech!"
    ]


def add_word_to_db(user_id: int, word: str):
    """Сохранить слово в БД (если не существует)."""
    translation = translate_text(word, "ru")
    c.execute("SELECT * FROM words WHERE user_id=? AND word=?", (user_id, word))
    row = c.fetchone()
    if not row:
        c.execute("""
        INSERT INTO words (user_id, word, translation, date_added)
        VALUES (?,?,?,?)
        """, (user_id, word, translation, datetime.datetime.now()))
        conn.commit()


def get_saved_words(user_id: int):
    c.execute("SELECT word, translation, times_correct, times_incorrect FROM words WHERE user_id=?", (user_id,))
    return c.fetchall()


def award_tokens(user_id: int, amount: float):
    c.execute("UPDATE users SET tokens = tokens + ? WHERE user_id=?", (amount, user_id))
    conn.commit()


def convert_tokens_to_eth(tokens: float) -> float:
    """Примерный курс: 1000 W2E = 1 ETH."""
    return tokens / 1000.0


def get_user_tokens(user_id: int) -> float:
    c.execute("SELECT tokens FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    return row[0] if row else 0.0


def record_chat_message(user_id: int, role: str, content: str, forwarded=False, audio=False):
    c.execute("""
    INSERT INTO chat_logs (user_id, role, content, timestamp, forwarded, audio)
    VALUES (?,?,?,?,?,?)
    """, (user_id, role, content, datetime.datetime.now(), forwarded, audio))
    conn.commit()


def is_forwarded_message(text: str) -> bool:
    """Определить, переслано ли сообщение."""
    triggers = ["forwarded", "переслано", ">"]
    lower = text.lower()
    return any(t in lower for t in triggers)


# ----------------------
#    Ollama Chat
# ----------------------
def ollama_chat(messages):
    """
    Вызываем Ollama локально.
    messages - список словарей [{"role": "system"/"user", "content": "..."}].
    """
    response = ollama.chat(model="qwen:7b-chat", messages=messages)
    # В зависимости от версии ollama, response["message"] может быть строкой
    # или dict. Предположим, что это просто строка-текст ответа.
    return response["message"]


# ----------------------
#  Оценка метрик (Pronunciation, Fluency...)
# ----------------------
def analyze_text_metrics(text: str) -> dict:
    """
    Примерно оцениваем 5 метрик и общий балл (0..100).
    Здесь для упрощения берем случайные числа,
    но вы можете подключить логику через T5/Grammar,
    speech-to-text confidence и т.д.
    """
    grammar_score = random.randint(50, 95)
    coherence_score = random.randint(50, 95)
    vocab_score = random.randint(50, 95)
    fluency_score = random.randint(50, 95)
    # Произношение (если речь - но сейчас просто рандом)
    pronunciation_score = random.randint(50, 95)

    overall = (grammar_score + coherence_score + vocab_score
               + fluency_score + pronunciation_score) // 5

    return {
        "pronunciation": pronunciation_score,
        "fluency": fluency_score,
        "vocab": vocab_score,
        "grammar": grammar_score,
        "coherence": coherence_score,
        "overall": overall
    }


# ---------------------- #
#       Интерфейс
# ---------------------- #
def main():
    st.title("💰 KON")

    user_id = get_current_user_id()

    # ------------- SIDEBAR -------------
    with st.sidebar:
        st.header("⚙️ Settings")

        # текущие данные
        c.execute("SELECT level, settings FROM users WHERE user_id=?", (user_id,))
        user_row = c.fetchone()
        current_level = user_row[0] if user_row else "A1"

        user_level = st.selectbox("Your English Level (CEFR)", ["A1", "A2", "B1", "B2", "C1", "C2"], index=0)
        score_format = st.radio("Scoring Format", ["100-scale", "IELTS", "TOEFL"], index=0)
        use_jokes = st.checkbox("Enable Jokes?", value=True)
        bot_role = st.radio("Bot Persona", ["Friendly Tutor", "Strict Teacher", "British Gentleman"], index=0)
        bot_voice = st.radio("Bot Voice", ["Male-US", "Female-UK", "Female-AU"], index=0)
        explanation_lang = st.selectbox("Language for Explanations", ["en", "ru", "es"], index=0)

        st.subheader("Limits")
        daily_audio_limit = st.number_input("Daily Audio Limit (sec)", value=300)
        text_limit = st.number_input("Text Message Limit (chars)", value=500)

        if st.button("Save Settings"):
            c.execute("UPDATE users SET level=? WHERE user_id=?", (user_level, user_id))
            conn.commit()
            st.success("Settings Saved.")

    # ---------------------- ТАБЫ ----------------------
    tab_chat, tab_progress, tab_quizzes, tab_earn = st.tabs(
        ["💬 Learn & Chat", "📈 Progress", "📚 Quizzes", "💰 Earn"]
    )

    # ============ TAB CHAT ============
    with tab_chat:
        st.subheader("Chat with AI Tutor (Ollama)")

        # Выведем прогресс-бары по 6 метрикам
        if "metrics" not in st.session_state:
            st.session_state["metrics"] = {
                "pronunciation": 0,
                "fluency": 0,
                "vocab": 0,
                "grammar": 0,
                "coherence": 0,
                "overall": 0
            }

        # 2 ряда колонок
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.progress(st.session_state["metrics"]["pronunciation"], text="Pronunciation")
        col2.progress(st.session_state["metrics"]["fluency"], text="Fluency")
        col3.progress(st.session_state["metrics"]["vocab"], text="Vocab")
        col4.progress(st.session_state["metrics"]["grammar"], text="Grammar")
        col5.progress(st.session_state["metrics"]["coherence"], text="Coherence")
        col6.progress(st.session_state["metrics"]["overall"], text="Overall")

        # История чата (последние 10)
        c.execute("SELECT role, content, timestamp FROM chat_logs WHERE user_id=? ORDER BY id DESC LIMIT 10",
                  (user_id,))
        chat_history = c.fetchall()[::-1]  # в хронологическом порядке

        for msg in chat_history:
            role_, content_, ts_ = msg
            st.markdown(f"**{role_}**: {content_}")

        user_input = st.text_input("Your Message")
        if st.button("Send to Tutor"):
            if not user_input.strip():
                st.warning("Please type something.")
            elif len(user_input) > text_limit:
                st.error("Message too long!")
            else:
                # Фильтр пересланных
                if is_forwarded_message(user_input):
                    st.warning("Forwarded message - no tokens awarded.")
                    record_chat_message(user_id, "User", user_input, forwarded=True)
                else:
                    # 1) Исправление ошибок (LanguageTool)
                    corrected = correct_text_with_languagetool(user_input)
                    # 2) Запись в чат
                    record_chat_message(user_id, "User", corrected, forwarded=False)

                    # 3) Оценим метрики (Grammar, Coherence, Vocab, etc.)
                    metrics_result = analyze_text_metrics(user_input)
                    st.session_state["metrics"] = metrics_result

                    # 4) Формируем запрос к Ollama
                    messages = [
                        {"role": "system", "content": f"You are a helpful English tutor with persona: {bot_role}."},
                        {"role": "user", "content": corrected}
                    ]
                    # Предположим, что ollama_chat(...) возвращает объект,
                    # из которого можно получить .content или .message
                    raw_response = ollama_chat(messages)

                    # Если raw_response - это объект, преобразуем его в строку
                    raw_response_str = str(raw_response)

                    if use_jokes:
                        raw_response_str += "\n(Here's a fun joke for you! 🏫)"

                    record_chat_message(user_id, f"Bot ({bot_role})", raw_response_str)

                    # Обновим интерфейс
                    st.rerun()

        st.write("---")
        st.write("#### Voice Message (Mock Demo)")
        audio_file = st.file_uploader("Upload .wav/.mp3", type=["wav", "mp3"])
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
            recognized_text = "(mock) Hello from audio!"
            recognized_text = fuzzy_normalize_speech(recognized_text)
            recognized_corrected = correct_text_with_languagetool(recognized_text)
            st.write(f"Recognized+Corrected: {recognized_corrected}")
            record_chat_message(user_id, "User (audio)", recognized_corrected, audio=True)
            st.success("Audio processed. (In real usage, integrate speech-to-text here.)")

        st.write("---")
        st.write("#### Click on a word to see details:")
        demo_sentence = "Hello traveler, would you like to purchase some souvenirs?"
        word_list = demo_sentence.split()
        colz = st.columns(len(word_list))
        for i, w in enumerate(word_list):
            if colz[i].button(w, key=f"demo_word_{i}"):
                with st.expander(f"Word Info: {w}"):
                    tr_ = translate_text(w, explanation_lang)
                    st.write(f"**Translation**: {tr_}")

                    syns = get_synonyms_en(w)
                    if syns:
                        st.write("**Synonyms**:", ", ".join(syns))

                    exs = get_examples_mock(w)
                    st.write("**Examples**:")
                    for e in exs:
                        st.write(f"- {e}")

                    ids_ = get_idioms_mock(w)
                    st.write("**Idioms/Phrases**:")
                    for i_ in ids_:
                        st.write("- " + i_)

                    audio_ = generate_tts(w, lang="en")
                    st.audio(audio_, format="audio/mp3")

                    if st.button(f"Save '{w}' to DB", key=f"save_{w}"):
                        add_word_to_db(user_id, w)
                        st.success(f"Saved '{w}'!")

    # ============ TAB PROGRESS ============
    with tab_progress:
        st.header("Progress Calendar")
        sel_date = st.date_input("Select Date", datetime.date.today())

        # Посмотрим, сколько слов выучено в этот день
        c.execute("SELECT COUNT(*) FROM words WHERE user_id=? AND date(date_added)=?", (user_id, sel_date))
        words_today = c.fetchone()[0]
        st.metric("Words Learned Today", words_today)

        # Сколько сообщений отправлено
        c.execute("SELECT COUNT(*) FROM chat_logs WHERE user_id=? AND date(timestamp)=?", (user_id, sel_date))
        msgs_today = c.fetchone()[0]
        st.metric("Chat Messages Today", msgs_today)

        # Покажем бар-чарт активности
        num_days = 30
        days_list = [datetime.date.today() - datetime.timedelta(days=i) for i in range(num_days)]
        days_list.reverse()

        msg_counts = []
        for d in days_list:
            c.execute("SELECT COUNT(*) FROM chat_logs WHERE user_id=? AND date(timestamp)=?", (user_id, d))
            msg_counts.append(c.fetchone()[0])

        df_chart = pd.DataFrame({"date": days_list, "messages": msg_counts})
        st.bar_chart(df_chart.set_index("date"))

        st.info("You could enhance this with a heatmap or advanced stats.")

    # ============ TAB QUIZZES ============
    with tab_quizzes:
        st.header("Your Quizzes")

        saved_words = get_saved_words(user_id)
        if not saved_words:
            st.warning("No saved words yet.")
        else:
            rnd_word = random.choice(saved_words)
            word_, transl_, cor_, incor_ = rnd_word
            st.write(f"**What is the translation of '{word_}'?**")

            # Варианты
            correct_ans = transl_
            dummy1 = "карандаш"
            dummy2 = "солнечный"
            opts = [correct_ans, dummy1, dummy2]
            random.shuffle(opts)

            pick = st.radio("Possible translations:", opts)
            if st.button("Check Answer"):
                if pick == correct_ans:
                    st.success("Correct! +10 tokens")
                    award_tokens(user_id, 10)
                    c.execute("UPDATE words SET times_correct=times_correct+1 WHERE user_id=? AND word=?",
                              (user_id, word_))
                else:
                    st.error("Incorrect!")
                    c.execute("UPDATE words SET times_incorrect=times_incorrect+1 WHERE user_id=? AND word=?",
                              (user_id, word_))
                conn.commit()

    # ============ TAB EARN ============
    with tab_earn:
        st.header("Earned Tokens")
        bal = get_user_tokens(user_id)
        st.metric("Your W2E Balance", f"{bal} W2E")

        # Показать эквивалент
        if score_format == "100-scale":
            st.write(f"Equivalent Score: {bal:.1f} / 100 scale (mock)")
        elif score_format == "IELTS":
            # 100 W2E ~ 9.0 IELTS
            ielts_score = (bal / 100) * 9.0
            st.write(f"IELTS approx: {ielts_score:.1f}")
        else:  # TOEFL
            toefl_score = (bal / 100) * 120
            st.write(f"TOEFL approx: {toefl_score:.0f}")

        if st.button("Convert to ETH"):
            eth = convert_tokens_to_eth(bal)
            st.success(f"Converted {bal} W2E to ~{eth:.5f} ETH (mock). Balance is reset to 0.")
            c.execute("UPDATE users SET tokens=0 WHERE user_id=?", (user_id,))
            conn.commit()

        st.write("You earn tokens by chatting, saving words, quizzes, etc.")

    st.write("---")
    st.caption("© KON")


if __name__ == "__main__":
    main()
