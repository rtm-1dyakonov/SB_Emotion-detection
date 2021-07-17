# Дипломный проект «Data Scientist. ML. Средний уровень (нейронные сети)»

Данный репозиторий включает в себя решение задачи разработки системы распознавания эмоций.           

Процесс был разбит на 2 этапа:
  1. *Разработка модели классификации эмоций*
  2. *Реализация real-time системы c использованием веб-камеры по выводу на экран текущей эмоции* 

## 1 этап (Google Colab) 
Изначально была спроектирована собственная [свёрточная сеть](https://colab.research.google.com/drive/1WEikgSCsdGmDOSYp6Xl27X4SwTSTqdSu?usp=sharing), которая вышла на плато по качеству валидации при значении ~0,45. 
Однако во избежание "изобретения велосипеда", было решенио использовать архитекутры из ModelZoo. Опираясь на сравнительную таблицу, 
![изображение](https://user-images.githubusercontent.com/65365762/126033483-0ec4d178-a900-4b18-adb0-d1b85f717a05.png)
взятую с интернет-ресурса [PyTorch for Beginners: Image Classification using Pre-trained models](https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/), я решил остановить выбор на 3-ёх архитектурах: VGG19, ResNet50 и ModileNet V2. Выбор обоснован как тем, что данные архитектуры обладают относительно малым инференсом и находятся в свободном доступе в пакете TensorFlow.  
Из-за специфики задачи использовался подход *fine-tuning*. Дело в том, что модели из "зоопарка" обучены на датасете ImageNet, где различия между классами гораздо более значительные нежели рахличия между эмоциями людей. Поэтому я решил разморозить все слои модели и обучить каждый, опираясь на таблицу (в случае данного кейса актуален 1-ый квадрант),
![изображение](https://user-images.githubusercontent.com/65365762/126033991-93dc22a5-9b8e-45b9-a2b6-03b264fe363d.png)
взятую из публикации [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751).

Максимальные значения метрики accuracy на валидационных данных получились следующие:
  * [VGG19](https://colab.research.google.com/drive/1o5vBPASvslthIGKJHk8J4Axy7oRGlKPk?usp=sharing) - 0,51; 
  * [ResNet50](https://colab.research.google.com/drive/15tMZfYb4x2lGH05Jz9gnwcojnQ-vt1CZ?usp=sharing) - 0,49;
  * [MobileNet V2](https://colab.research.google.com/drive/1LHDGYcfnDxggoQih4c3tgC-ylqjyiIu4?usp=sharing) - 0,46.

  Данные значения соответствуют обучению с аугментацией данных. Без неё метрика accuracy в среднем была ниже на 5-6%. 
  Соответственно, основой для продолжения разработки стала модель с архитекутрой VGG19. 
  
Изначально после обучения модель НЕ соответствовала требованиям по времени инференса (**58-66мс** на 1 кадр против максимально допустимых **33мс**). Для алгоритмического ускорения был использован пакет TensorRT с собственным "ифнвренс-движком", который относительно недавно был встроен непосредственно в библиотеку TensorFlow. После подобной модификации модели скорость инференса к знаениям **5-10мс** на 1 кадр.

[Отдельный блокнот](https://colab.research.google.com/drive/1rLKzOY3qA8eAstIIn5bdA0uxqMaDQTl-?usp=sharing) служил для формирования файла на платформу Kaggle. Итоговый результат посылки получился следующий:
![изображение](https://user-images.githubusercontent.com/65365762/126034212-fa823f25-d3b9-4ce9-9234-f7d85dc46d65.png)
На платформе присутствует 2 попытки отправки файла submission.csv из-за того, что в первый раз словарь с названиями классов был сформирован неправильно: нужно было пометить их в алфавитном порядке, а не в том, как они расположены на Google Drive  
![GIF](https://media.giphy.com/media/GDEkCw4R52oRG/source.gif)  

## 2 этап (Visual Studio Code) 

    




