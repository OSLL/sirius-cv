Запуск скрипта через консоль:
```bash
pip install -r requirements.txt
# python <название скрипта> --input <путь входному видео> --output <путь, куда сохранить выходное видео> --sings <путь, где находятся изображения знаков>
# Пример (для Linux):
python3.8 test.py --input input.mp4 --output output.avi --sings sings

# Для Windows
python test.py --input input.mp4 --output output.avi --sings sings 
```
> Версия python: **3.8**
> 
> Для выходного видео нужно использовать **только** формат *.avi*
> 
> Путь до знаков нужно указывать относительно скрипта:
> > Например: `test_signs`, если папка со знаками лежит в той же директории,
> что и скрипт.

Пример детекции знаков:

![Детекция знаков, git изображение](output.gif)