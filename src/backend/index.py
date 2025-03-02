import os
from dotenv import load_dotenv

import telebot

from src.rag_components.assistant import Assistant


TOKEN = os.getenv('TELEBOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

assistant = Assistant()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
    Привет! Я ИИ-ассистент для игры Dungeons And Dragons. Задай мне любой вопрос по игре!
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(content_types=['text'])
def handle_message(message):
    try:
        user_query = message.text
        response = assistant.answer(user_query)
        print('response len', len(response))
        # telegram allows 4096 maximum length in one message
        chunk_size = 4000
        chunks = [response[i:i+chunk_size] for i in range(0, len(response), chunk_size)]
        for message_part in chunks:
            bot.send_message(
                message.chat.id, 
                message_part
            )
    except Exception as e:
        bot.reply_to(message, f"⚡ An error occurred: {str(e)}")

if __name__ == '__main__':
    print("Bot is running...")
    bot.infinity_polling()
