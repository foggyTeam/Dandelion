import os

from dotenv import load_dotenv
from telebot import TeleBot

from src.model import predict_flower
from src.utils import colored_print

load_dotenv()
BOT_TOKEN = os.environ['BOT_TOKEN']
bot = TeleBot(BOT_TOKEN)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[0].file_id)
    file = bot.download_file(file_info.file_path)

    flower, confidence = predict_flower(file)
    colored_print(f'\nPredicted {flower} with confidence {confidence:.2f}', 'c')

    if confidence >= 0.8:
        answer = f'Это же {flower}!'
    elif confidence >= 0.4:
        answer = f'Возможно, {flower}? (уверен на {(confidence * 100):.0f}%)'
    else:
        answer = f'Я не знаю, что это за цветок. Может быть, {flower}? (уверен на {(confidence * 100):.0f}%)'

    bot.reply_to(message, answer)
