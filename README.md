# Дипломный проект «Data Scientist. ML. Средний уровень (нейронные сети)»

Данный репозиторий включает в себя решение задачи разработки системы распознавания эмоций.           

Процесс был разбит на 2 этапа:
  1. *Разработка модели классификации эмоций*
  2. *Реализация real-time системы c использованием веб-камеры по выводу на экран текущей эмоции* 

## 1 этап (Google Colab) 
Изначально была спроектирована собственная [свёрточная сеть](https://colab.research.google.com/drive/1WEikgSCsdGmDOSYp6Xl27X4SwTSTqdSu?usp=sharing) (**Custom.ipynd**), которая вышла на плато по качеству валидации при значении ~0,45. 
Однако во избежание "изобретения велосипеда", было решенио использовать архитекутры из ModelZoo. Опираясь на сравнительную таблицу, 
![изображение](https://user-images.githubusercontent.com/65365762/126033483-0ec4d178-a900-4b18-adb0-d1b85f717a05.png)
взятую с интернет-ресурса [PyTorch for Beginners: Image Classification using Pre-trained models](https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/), я решил остановить выбор на 3-ёх архитектурах: VGG19, ResNet50 и ModileNet V2. Выбор обоснован как тем, что данные архитектуры обладают относительно малым инференсом и находятся в свободном доступе в пакете TensorFlow.  
Из-за специфики задачи использовался подход *fine-tuning*. Дело в том, что модели из "зоопарка" обучены на датасете ImageNet, где различия между классами гораздо более значительные нежели рахличия между эмоциями людей. Поэтому я решил разморозить все слои модели и обучить каждый, опираясь на таблицу (в случае данного кейса актуален 1-ый квадрант),
![изображение](https://user-images.githubusercontent.com/65365762/126033991-93dc22a5-9b8e-45b9-a2b6-03b264fe363d.png)
взятую из публикации [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751).

Максимальные значения метрики accuracy на валидационных данных получились следующие:
  * [VGG19](https://colab.research.google.com/drive/1o5vBPASvslthIGKJHk8J4Axy7oRGlKPk?usp=sharing) - 0,51 (**VGG_19_(AUG).ipynb**); 
  * [ResNet50](https://colab.research.google.com/drive/15tMZfYb4x2lGH05Jz9gnwcojnQ-vt1CZ?usp=sharing) - 0,49 (**ResNet50.ipynb**);
  * [MobileNet V2](https://colab.research.google.com/drive/1LHDGYcfnDxggoQih4c3tgC-ylqjyiIu4?usp=sharing) - 0,46 (**MobileNet_V2_(simple_+_AUG).ipynb**).

  Данные значения соответствуют обучению с аугментацией данных. Без неё метрика accuracy в среднем была ниже на 5-6%. 
  Соответственно, основой для продолжения разработки стала модель с архитекутрой VGG19. 
  
Изначально после обучения модель НЕ соответствовала требованиям по времени инференса (**58-66мс** на 1 кадр против максимально допустимых **33мс**). Для алгоритмического ускорения был использован пакет TensorRT с собственным "ифнвренс-движком", который относительно недавно был встроен непосредственно в библиотеку TensorFlow. После подобной модификации модели скорость инференса приблизиласт к значениям **5-10мс** на 1 кадр.

[Отдельный блокнот](https://colab.research.google.com/drive/1rLKzOY3qA8eAstIIn5bdA0uxqMaDQTl-?usp=sharing) (**Kaggle_format.ipynb**) служил для формирования файла на платформу Kaggle. Итоговый результат посылки получился следующий:
![изображение](https://user-images.githubusercontent.com/65365762/126034212-fa823f25-d3b9-4ce9-9234-f7d85dc46d65.png)
На платформе присутствует 2 попытки отправки файла submission.csv из-за того, что в первый раз словарь с названиями классов был сформирован неправильно: нужно было пометить их в алфавитном порядке, а не в том, как они расположены на Google Drive  
![GIF](https://media.giphy.com/media/GDEkCw4R52oRG/source.gif)  

## 2 этап (Visual Studio Code) 

Cкрипт для классификации эмоций с использованием веб-камеры *main.py* написан в Visual Studio Code. 
Для использования веб-камеры применяется функционал библиотеки *OpenCV*. 

![GIF-2](https://media.giphy.com/media/87cLAJUaosOA3FQKKJ/giphy.gif)  

Есть 2 способа запуска системы на вашем устройстве:
  1. Скачать архив по [ссылке](https://drive.google.com/file/d/1GdflGrwuEtljz3CpnuJugINq47E-I_uZ/view?usp=sharing), распаковать на своё устройство и запустить файл main.exe. После небольшого ожидания должна запуститься камера  
  2. Руководствоваться следующими указаниями (в этом случае для воспроизведения решения на Вашем устройстве **должен** быть установлен Python. Для установки см. [ссылку](https://thecode.media/py-install/)).

Шаги для использования системы распознавания эмоций:
  1. Скачать *haarcascade_frontalface_default.xml* из данного репозитория. 
  2. Скачать *модель по классификации эмоций* одним из 2-ух следующих способов:
  * Скачать и распаковать архив с моделью по классификации эмоций по [ссылке](https://drive.google.com/file/d/162YQlfzF2MPvYdqpTkdpVDy4Kf4mGPoS/view?usp=sharing)
  * Скачать модель, обученную самостоятельно и ускоренную с помощью TensorRT (в таком случае модель должна обучаться на изображениях (128, 128, 3) для воспроизводимости скрипта)
  3. Скачать файл *main.py*
  4. Скрипт *main.py*, модель *haarcascade_frontalface_default.xml* и папка с *моделью по классификации эмоций* (saved_model_trt либо Ваша модель) должны лежать в 1 директории
  ![изображение](https://user-images.githubusercontent.com/65365762/126067086-0daee382-d728-4cc8-92d1-2c6cca1f8834.png)
  5. В папке с файлом *main.py* открыть командную строчку и прописать **py main.py**
  6. Если у Вас не установлен Keras или cv2 (OpenCV), то понадобится прописать **pip install keras** или **pip install opencv-python** соответственно, а после повторить шаг 5.

## Ссылки для скачивания файлов

  * [test_kaggle.zip](https://drive.google.com/file/d/1kGxkvH8c9WC1AKKZpXOmEbesS0EiZ7B7/view?usp=sharing); 
  * [train.zip](https://drive.google.com/file/d/1JcZGqGFubGi_7W-RCO1StPhvw90Qlfx0/view?usp=sharing);
  * [VGG19 model (21 эпоха)](https://drive.google.com/file/d/1C0UlrasAdX5xzTgDIz-kn8naIVFO8W01/view?usp=sharing);
  * [saved_model_trt](https://drive.google.com/file/d/162YQlfzF2MPvYdqpTkdpVDy4Kf4mGPoS/view?usp=sharing) (VGG19 model (21 эпоха) + TensorRT).



