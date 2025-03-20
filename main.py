from src.bot import bot
from src.utils import colored_print


def main():
    colored_print('\nStarting Dandelion!\n', 'g')

    while True:
        try:
            bot.polling()
        except RuntimeError:
            colored_print('An error occurred. Restarting...\n', 'y')


if __name__ == '__main__':
    main()
