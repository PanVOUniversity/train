# Быстрый старт обучения на одноклассовом COCO датасете

Эта папка содержит вспомогательные скрипты для обучения самого быстрого встроенного модели Detectron2 для сегментации экземпляров (Mask R-CNN R50-FPN 1× schedule) на пользовательском датасете в формате COCO с одним классом, закодированным в формате RLE.

## Предварительные требования

Перед обучением необходимо сгенерировать COCO датасет с помощью пайплайна генерации данных:

1. Генерация HTML страниц: `data-generation/html_generator.py`
2. Рендеринг скриншотов: `data-generation/playwright_render.py`
3. Генерация метаданных: `data-generation/make_masks.py`
4. Конвертация в формат COCO: `data-generation/coco_converter.py`

Скрипт `coco_converter.py` создает следующую структуру:

```
data/coco/
  train/                       # PNG изображения для обучающей выборки
  val/                         # PNG изображения для валидационной выборки
  annotations/
    instances_train.json       # Аннотации в формате COCO с RLE `segmentation`
    instances_val.json         # Аннотации в формате COCO для валидации
```

## 1. Подготовка структуры датасета

Функция `register_single_class_coco` ожидает, что все изображения находятся в одной директории (стандартный формат COCO), но `coco_converter.py` создает отдельные директории `train/` и `val/`. У вас есть два варианта:

### Вариант A: Объединить изображения в одну директорию (Рекомендуется)

Скопируйте все изображения в одну директорию `images/`:

```bash
# Создать директорию images
mkdir -p data/coco/images

# Скопировать изображения для обучения
cp data/coco/train/*.png data/coco/images/

# Скопировать изображения для валидации (у них разные имена, поэтому конфликтов не будет)
cp data/coco/val/*.png data/coco/images/
```

Итоговая структура должна быть:

```
data/coco/
  images/                      # Все изображения (train + val)
  annotations/
    instances_train.json       # Ссылается на изображения в images/
    instances_val.json         # Ссылается на изображения в images/
```

## 1. Регистрация датасета

Перед обучением зарегистрируйте разделы датасета в Detectron2:

```bash
python train/register_dataset.py \
  --dataset-root data/coco \
  --images-subdir images \
  --train-json annotations/instances_train.json \
  --val-json annotations/instances_val.json \
  --thing-classes target
```

**Примечание:** Параметр `--images-subdir` должен указывать на директорию, содержащую все изображения (по умолчанию `images`). Оба JSON файла (train и val) ссылаются на изображения из этой единственной директории.

Скрипт регистрирует имена датасетов `my_coco_train` / `my_coco_val` (можно переопределить с помощью `--train-name` / `--val-name`) и присваивает имя класса `target`. Внутри используется `register_coco_instances`, поэтому результирующие загрузчики полностью поддерживают RLE маски, когда `cfg.INPUT.MASK_FORMAT = "bitmask"`.

## 3. Запуск обучения

`train/train_single_class.py` — это тонкая обертка вокруг `DefaultTrainer` Detectron2, которая выбирает легковесную конфигурацию Mask R-CNN R50-FPN 1× и предоставляет флаги, настроенные для одного класса. Пример команды:

```bash
python train/train_single_class.py \
  --dataset-root data/coco \
  --images-subdir images \
  --output-dir output/my_target \
  --ims-per-batch 4 \
  --max-iter 8000 \
  --lr-steps 6000 7500 \
  --num-gpus 1
```

**Важно:** Убедитесь, что используете те же пути `--dataset-root` и `--images-subdir`, которые использовали при регистрации датасета в шаге 2.

### Ключевые значения по умолчанию

- Файл конфигурации: `configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml`
- Предобученные веса: чекпоинт `mask_rcnn_R_50_FPN_1x` из модели Detectron2 model zoo
- `MODEL.ROI_HEADS.NUM_CLASSES = 1`
- `INPUT.MASK_FORMAT = "bitmask"` для сегментации в формате RLE COCO
- Имена датасетов: `my_coco_train` / `my_coco_val`
- Имя класса: `target`

### Параметры настройки

Переопределите любые из этих значений по умолчанию по необходимости:

- `--thing-classes`: Изменить имена классов (по умолчанию: `target`)
- `--train-dataset` / `--val-dataset`: Изменить имена датасетов (по умолчанию: `my_coco_train` / `my_coco_val`)
- `--base-lr`: Скорость обучения (по умолчанию: 0.00025)
- `--max-iter`: Максимальное количество итераций (по умолчанию: 5000)
- `--lr-steps`: Шаги уменьшения скорости обучения (по умолчанию: 3500 4500)
- `--checkpoint-period`: Сохранять чекпоинт каждые N итераций (по умолчанию: 1000)
- `--eval-period`: Запускать оценку каждые N итераций (по умолчанию: 500)
- `--ims-per-batch`: Изображений в батче (по умолчанию: 4)

## 3. Проверка и мониторинг

### Визуализация аннотаций

Проверьте аннотации, чтобы убедиться, что маски/классы загружаются правильно:

```bash
python tools/visualize_data.py \
  --source annotation \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --dataset my_coco_train \
  --output-dir output/visualizations
```

### Мониторинг обучения

Логи обучения/оценки и метрики сохраняются в `OUTPUT_DIR` (по умолчанию `output/my_target`). Используйте TensorBoard для мониторинга:

```bash
tensorboard --logdir output/my_target
```

Метрики также сохраняются как JSON файлы в выходной директории для программного доступа.

### Запуск только валидации

Для оценки обученной модели без обучения:

```bash
python train/train_single_class.py \
  --dataset-root data/coco \
  --images-subdir images \
  --output-dir output/my_target \
  --eval-only \
  --resume \
  --weights output/my_target/model_final.pth
```

### Запуск инференса на изображении

Для запуска обученной модели на вашем изображении и сохранения масок:

```bash
python train/inference.py \
  --image path/to/your/image.png \
  --weights output/my_target/model_final.pth \
  --output-dir output/inference \
  --num-classes 1 \
  --thing-classes frame \
  --confidence-threshold 0.5
```

**Параметры:**
- `--image`: Путь к входному изображению
- `--weights`: Путь к обученной модели (`model_final.pth`)
- `--output-dir`: Директория для сохранения результатов (по умолчанию: `output/inference`)
- `--num-classes`: Количество классов (по умолчанию: 1)
- `--thing-classes`: Имена классов (по умолчанию: `frame`)
- `--confidence-threshold`: Порог уверенности для детекций (по умолчанию: 0.5)
- `--device`: Устройство для инференса (`cpu` или `cuda`). По умолчанию определяется автоматически

**Запуск на CPU:**
```bash
python train/inference.py \
  --image path/to/image.png \
  --weights model_final.pth \
  --device cpu \
  --output-dir output/inference
```

**Результаты сохраняются в:**
- `output/inference/masks/` — отдельные маски для каждого обнаруженного объекта (`image_mask_0.png`, `image_mask_1.png`, ...)
- `output/inference/masks/image_combined_mask.png` — объединенная цветная маска всех объектов
- `output/inference/image_visualization.png` — визуализация с наложенными масками и bounding boxes

### Просмотр метрик обученной модели

Для просмотра метрик обучения и оценки:

```bash
python train/view_metrics.py --output-dir output/my_target
```

Скрипт покажет:
- Метрики обучения (loss, learning rate, время итерации и т.д.)
- Статистику по всему процессу обучения
- Результаты оценки (AP, AP50, AP75 и т.д.), если они доступны
- Информацию о модели

Для экспорта метрик в CSV:

```bash
python train/view_metrics.py --output-dir output/my_target --export-csv metrics.csv
```

**Альтернативные способы просмотра метрик:**

1. **Прямой просмотр JSON файла:**
   ```bash
   # Просмотр последних метрик
   tail -n 5 output/my_target/metrics.json | jq .
   
   # Просмотр конкретной метрики (например, loss_mask)
   cat output/my_target/metrics.json | jq -r '.loss_mask'
   ```

2. **TensorBoard (если доступен):**
   ```bash
   tensorboard --logdir output/my_target
   ```

3. **Просмотр лога обучения:**
   ```bash
   # Поиск метрик оценки в логе
   grep "copypaste:" output/my_target/log.txt
   ```

## Полный пример workflow

Вот полный пример от генерации данных до обучения:

```bash
# 1. Генерация HTML страниц
python data-generation/html_generator.py \
  --output-dir data/pages \
  --num-pages 100

# 2. Рендеринг скриншотов
python data-generation/playwright_render.py \
  --input-dir data/pages \
  --output-dir data/screenshots \
  --meta-dir data/meta

# 3. Генерация масок (если необходимо)
python data-generation/make_masks.py \
  --meta-dir data/meta \
  --output-dir data/masks

# 4. Конвертация в формат COCO (автоматически создаст папку images/)
python data-generation/coco_converter.py \
  --meta-dir data/meta \
  --screenshots-dir data/screenshots \
  --output-dir data/coco

# 5. Регистрация датасета
python train/register_dataset.py \
  --dataset-root data/coco \
  --images-subdir images \
  --thing-classes target

# 6. Обучение модели
python train/train_single_class.py \
  --dataset-root data/coco \
  --images-subdir images \
  --output-dir output/my_target \
  --num-gpus 1
```

## Решение проблем

### Изображения не найдены

Если вы видите ошибки о недостающих изображениях:

- Убедитесь, что все изображения находятся в директории `images/` (или в директории, указанной в `--images-subdir`)
- Проверьте, что имена файлов изображений в JSON аннотациях совпадают с фактическими именами файлов в директории images
- Убедитесь, что пути относительны к `dataset-root`

### Ошибки регистрации датасета

- Убедитесь, что `--dataset-root` указывает на директорию, содержащую `images/` и `annotations/`
- Проверьте существование JSON файлов: `annotations/instances_train.json` и `annotations/instances_val.json`
- Убедитесь, что директория images существует и содержит PNG файлы

### Ошибки обучения

- Убедитесь, что датасет зарегистрирован перед началом обучения
- Проверьте, что `--train-dataset` и `--val-dataset` совпадают с именами, использованными при регистрации
- Убедитесь, что GPU доступен, если используете `--num-gpus > 0`

### Ошибка `AttributeError: module 'distutils' has no attribute 'version'`

Если вы видите ошибку о недоступном `distutils.version`, это может происходить по нескольким причинам:

1. **Python 3.12+**: `distutils` был удален из стандартной библиотеки Python
2. **Python 3.8-3.11 с setuptools >= 65**: Начиная с версии 65, `setuptools` больше не включает `distutils`

**Решение:** Установите совместимую версию `setuptools` (версия < 65):

```bash
pip install 'setuptools<65'
```

Или, если нужна последняя версия `setuptools`, можно установить пакет `setuptools-scm`, который может помочь:

```bash
pip install 'setuptools<65' setuptools-scm
```

**Важно:** Если ошибка все еще возникает даже после установки `setuptools<65`, скрипт автоматически продолжит работу без TensorBoard логирования (метрики все равно будут сохраняться в JSON формате в файле `metrics.json`). Это не критично для обучения - TensorBoard используется только для визуализации метрик.

### Запуск модели на Windows с CPU

Для запуска обученной модели на Windows компьютере с CPU см. подробный гайд: [WINDOWS_CPU_SETUP.md](../WINDOWS_CPU_SETUP.md)

**Краткая инструкция:**

1. Скопируйте модель с сервера:
   ```bash
   scp -i .\ssh\key root@91.236.199.226:/tmp/model_final.pth ./
   ```

2. Установите Detectron2 (рекомендуется через WSL2):
   ```bash
   # В WSL2
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

3. Запустите инференс:
   ```bash
   python train/inference.py --image image.png --weights model_final.pth --device cpu
   ```

**Требования к ресурсам:**
- RAM: 4-8 GB (рекомендуется 8+ GB)
- CPU: Современный процессор (Intel Core i5/i7 или AMD Ryzen 5/7)
- Время обработки: 5-30 секунд на изображение (зависит от размера)

Следуя этим шагам, вы запустите самую быструю версию Mask R-CNN, которая все еще поддерживает маски экземпляров, минимизируя время обучения при соблюдении требования одноклассовой сегментации COCO.
