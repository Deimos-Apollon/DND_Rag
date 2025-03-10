# Описание проекта

Ассистент для игры Dungeons And Dragons (5-ая редакция правил). Использует RAG для извлечения релевантного контекста и последующей генерации ответа. Интерфейс взаимодействия происходит через телеграм-бота @dndrag_bot.

# Данные

* [Книга правил Мастера](https://www.dungeonsanddragons.ru/bookfull/5ed/5e%20Dungeon%20Masters%20Guide%20-%20%D0%A0%D1%83%D0%BA%D0%BE%D0%B2%D0%BE%D0%B4%D1%81%D1%82%D0%B2%D0%BE%20%D0%9C%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B0%20RUS.pdf) (рус.)
* [Книга правил Игрока](https://dungeonsanddragons.ru/bookfull/5ed/5e%20Players%20Handbook%20-%20%D0%9A%D0%BD%D0%B8%D0%B3%D0%B0%20%D0%B8%D0%B3%D1%80%D0%BE%D0%BA%D0%B0%20RUS.pdf) (рус.)

# Переменные окружения

Нужно создать файл *.env* (пример файла: *.env_example*)и поместить в него две переменные окружения:

1. *gigachat_credentials.* Ключ API для доступа к ГигаЧат. Можно использовать другую LLM, тогда её нужно передавать в класс Assistant при инициализации.
2. *TELEBOT_TOKEN.* Токен для телеграм бота.

# Инструкция к запуску

1. Развернуть локально репозиторий `git clone https://github.com/Deimos-Apollon/DND_Rag`
2. Создать виртуальное окружение (рекомендуется) `python -m venv .venv`
3. Установить зависимости `pip install -r requirements.txt`
4. Добавить путь к директории (чтобы внутренние модули были видны) `export PYTHONPATH="${PYTHONPATH}:<локальный путь к репозиторию>/DnDRAG/src"`
5. Запуск бота через `python src/backend/index.py`

# Тестирование

Качество ответов исследовалось по трём метрикам:

| Faithfulness score | Relevancy | Correctness |
| ------------------ | --------- | ----------- |
| 0.25               | 0.71      | 0.73        |

Низкий faitfulness score скорее всего вызван разной длиной ответов и степенью их подробности. В качестве "идеальных" ответов использовались сгенерированные моделью DeepSeek r1, которые были достаточно сжатыми. Рассматриваемый Ассистент возвращает более подробный ответ, опираясь на контекст.

Для получаения метрики Relevancy успехом считалось, если для запроса релевантен хотя бы один из использованных в контексте нодов.

Метрика Correctness отражает насколько похожи "идеальные" ответы и полученные моделью в смысле ответа. Использовалось среднее количество успешно пройденных запросов.
