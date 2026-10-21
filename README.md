🌲 Forest Telegram Bot (RAG Assistant)
Forest Telegram Bot — це інтелектуальний Telegram-бот на базі архітектури RAG (Retrieval-Augmented Generation), призначений для відповідей на запитання щодо лісової документації та порад. Бот шукає відповідну інформацію в локальних та завантажених користувачами документах за допомогою векторного пошуку і формує точні відповіді українською мовою з посиланням на джерела.
---
🚀 Основні можливості
- Пошук за документами (RAG): Використовує модель семантичного занурення (`sentence-transformers/intfloat/multilingual-e5-small`) для знаходження найрелевантніших фрагментів тексту.
- Генерація відповідей через Groq LLM: Відповідає суворо за знайденим контекстом за допомогою моделей `openai/gpt-oss-120b` або `openai/gpt-oss-20b`.
- Підтримка багатьох форматів: Обробка `.pdf`, `.docx`, `.txt` та `.md` файлів.
- Розпізнавання тексту (OCR): Автоматичне сканування сканованих PDF-файлів за допомогою `Tesseract OCR` (українська мова).
- Синхронізація з Google Drive: Завантаження та синхронізація базових документів із папки Google Drive або локальної директорії `base_docs`.
- Завантаження користувацьких порад: Користувачі можуть надсилати власні документи у приватних повідомленнях боту. Бот перевіряє тематику документа, запитує автора та сферу застосування (рукавичка / частина лісу / весь ліс).
- Посилання на джерела: До кожної відповіді бот надає інтерактивне посилання на використане джерело (Google Drive або файл користувача).
- Контекстний діалог: Збереження історії останніх повідомлень для уточнення запитань.
---
🛠 Технологічний стек
Мова: Python 3.10+
Фреймворк для бота: `aiogram 3.x`
LLM API: `AsyncGroq`
Ембединги та векторний пошук: `sentence-transformers`, `NumPy`
Обробка документів: `PyMuPDF (fitz)`, `pypdf`, `python-docx`, `Pillow`, `pytesseract`
Контейнеризація: `Docker`, `Docker Compose`
---
📁 Структура проєкту
```text
.
├── bot.py                # Основний скрипт бота (обробка повідомлень, RAG, пошук)
├── requirements.txt      # Залежності проєкту
├── Dockerfile            # Інструкція для збірки Docker-контейнера
├── docker-compose.yml    # Конфігурація Docker Compose
├── env.example           # Приклад налаштувань середовища (.env)
├── .gitignore            # Файли для ігнорування у Git
└── .dockerignore         # Файли для ігнорування у Docker
```
---
⚙️ Налаштування та запуск
1. Клонування репозиторію
```bash
git clone https://github.com/Rar571/forest-telegram-bot.git
cd forest-telegram-bot
```
2. Налаштування змінних оточення
Створіть файл `.env` на основі `env.example`:
```bash
cp env.example .env
```
Заповніть потрібні ключі у `.env`:
```env
TELEGRAM_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
DRIVE_API_KEY=your_google_drive_api_key 
DRIVE_FOLDER_ID=your_google_drive_folder_id 
```
---
🐳 Запуск через Docker (Рекомендовано)
Найпростіший спосіб розгорнути бота — скористатися Docker Compose:
```bash
docker-compose up -d --build
```
Зупинити бота:
```bash
docker-compose down
```
---
💻 Локальний запуск (без Docker)
Встановлення системних залежностей (для OCR)
Для роботи розпізнавання тексту з PDF потрібен встановлений `Tesseract OCR` з українським мовним пакетом:
Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-ukr libgeos-dev`
macOS: `brew install tesseract tesseract-lang`
Кроки запуску:
Створіть та активуйте віртуальне середовище:
```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/macOS
   # або venv\Scripts\activate  # Для Windows
   ```
Встановіть Python-пакети:
```bash
   pip install -r requirements.txt
   ```
Запустіть бота:
```bash
   python bot.py
   ```
---
💬 Команди бота в Telegram
`/start` — Привітання та коротка інструкція з користування.
`/reset` — Очистити історію поточного діалогу.
`/cancel` — Скасувати процес завантаження документа.
`/sync` — Синхронізувати базову документацію з Google Drive або локальної папки.
`/debug <питання>` — Показати схожість завантажених документів за вектором.
