from src.bot import bot
from src.utils import colored_print


def main():
    colored_print('\nStarting Dandelion!\n', 'g')

    bot.polling()


if __name__ == '__main__':
    main()
