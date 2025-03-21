# Dandelion

_Телеграм-бот, распознающий тип цветка по изображению._

[GitHub](https://github.com/foggyTeam/Dandelion)

[Google Slides](https://docs.google.com/presentation/d/1HIh1DOSdzfC5lbUNx0QXvmyz-Wi5GFagbqdCL_H5E-A/edit?usp=sharing)

## О проекте

Нейросеть распознает 9 видов цветов:

- орхидея
- лаванда
- подсолнух
- лилия
- лотос
- одуванчик
- ромашка

[Телеграм-бот](https://t.me/dandelion_ai_bot)

## Обучение и выбор моделей

### Датасет

В результате работы нам пришлось создать собственный датасет, где на каждый класс
приходится суммарно 1000 изображений. В основу его легли датасеты с Kaggle:

- National Flowers
- 5 Flower Types Classification Dataset
- Flowers Recognition
- 🌸 | Flowers

Данные были взяты из этих датасетов, однако мусорная информация была отфильтрована вручную для каждого класса, чтобы
избежать, например, нерелевантных для обучения и тестирования фотографий букетов с несколькими классами цветов.

[Итоговый датасет](https://www.kaggle.com/datasets/iciicifur/national-flowers-vdandelion)

### Модели

_В основе телеграм-бота лежит модель ResNet50._

Мы обучили 4 модели:

- CNN (собственная простая модель)
- ResNet50
- MobileNet v2
- EfficientNet b0

Наилучшие результаты (91% accuracy) показали ResNet50 и MobileNet. Мы выбрали первую, так как она лучше распознавала
одуванчики, что критично для названия нашего проекта.

[Ноутбук](https://www.kaggle.com/code/iciicifur/dandelion)