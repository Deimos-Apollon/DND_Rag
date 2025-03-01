import os
from dotenv import load_dotenv

import telebot

from src.rag_components.agent import Agent


TOKEN = os.getenv('TELEBOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

assistant = Agent()

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
    Привет! Я ИИ-ассистент для игры Dungeons And Dragons. Задай мне вопрос по этой игре!.
    """
    bot.reply_to(message, welcome_text)

@bot.message_handler(content_types=['text'])
def handle_message(message):
    try:
        user_query = message.text
        response = assistant.answer(user_query)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"⚡ An error occurred: {str(e)}")

if __name__ == '__main__':
    print("Bot is running...")
    bot.infinity_polling()
