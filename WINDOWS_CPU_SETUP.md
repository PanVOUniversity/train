# Гайд по установке и запуску модели на Windows с CPU

Этот гайд описывает, как скопировать обученную модель на Windows компьютер и запустить инференс на CPU.

## Содержание

1. [Копирование модели с сервера](#копирование-модели-с-сервера)
2. [Установка Detectron2 на Windows](#установка-detectron2-на-windows)
3. [Запуск инференса на CPU](#запуск-инференса-на-cpu)
4. [Требования к ресурсам](#требования-к-ресурсам)
5. [Решение проблем](#решение-проблем)

---

## Копирование модели с сервера

### Шаг 1: Скопируйте модель и конфигурацию

На вашем Windows компьютере (PowerShell или CMD):

```bash
# Скопируйте модель
scp -i .\ssh\key root@91.236.199.226:/tmp/inference/model_final.pth ./

# Или если модель находится в контейнере, сначала скопируйте на хост:
# На сервере (вне контейнера):
docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/my_target/model_final.pth /tmp/
docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/my_target/config.yaml /tmp/

# Затем с Windows:
scp -i .\ssh\key root@91.236.199.226:/tmp/model_final.pth ./
scp -i .\ssh\key root@91.236.199.226:/tmp/config.yaml ./
```

### Шаг 2: Скопируйте конфигурационный файл

```bash
# Скопируйте конфигурационный файл из репозитория
scp -i .\ssh\key root@91.236.199.226:/tmp/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml ./
```

Или скачайте напрямую с GitHub:
```bash
# В PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml" -OutFile "mask_rcnn_R_50_FPN_1x.yaml"
```

---

## Установка Detectron2 на Windows

### Вариант 1: Использование WSL2 (Рекомендуется)

**WSL2 (Windows Subsystem for Linux) - самый простой способ:**

1. **Установите WSL2:**
   ```powershell
   # В PowerShell от имени администратора
   wsl --install
   ```

2. **Установите Ubuntu из Microsoft Store**

3. **В WSL установите зависимости:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev git
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip3 install opencv-python pillow
   ```

4. **Установите Detectron2:**
   ```bash
   pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

### Вариант 2: Нативный Windows (Сложнее)

**Требования:**
- Python 3.8-3.11 (Python 3.12+ может иметь проблемы)
- Visual Studio Build Tools с C++ компилятором
- CMake

**Установка:**

1. **Установите Python:**
   - Скачайте с https://www.python.org/downloads/
   - При установке отметьте "Add Python to PATH"

2. **Установите Visual Studio Build Tools:**
   - Скачайте: https://visualstudio.microsoft.com/downloads/
   - Выберите "Build Tools for Visual Studio"
   - Установите "Desktop development with C++"

3. **Установите CMake:**
   - Скачайте: https://cmake.org/download/
   - Или через pip: `pip install cmake`

4. **Установите PyTorch (CPU версия):**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Установите зависимости:**
   ```powershell
   pip install opencv-python pillow numpy
   ```

6. **Установите Detectron2:**
   ```powershell
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

   **Примечание:** Установка может занять 10-30 минут, так как компилируется из исходников.

---

## Запуск инференса на CPU

### Шаг 1: Подготовьте скрипт для CPU

Создайте файл `inference_cpu.py` на основе `train/inference.py`, но с явным указанием CPU:

```python
# В setup_cfg добавьте:
cfg.MODEL.DEVICE = "cpu"
```

Или используйте готовый скрипт (см. ниже).

### Шаг 2: Запустите инференс

```bash
python inference_cpu.py \
  --image path/to/your/image.png \
  --weights model_final.pth \
  --config-file mask_rcnn_R_50_FPN_1x.yaml \
  --output-dir output/inference \
  --num-classes 1 \
  --thing-classes frame \
  --confidence-threshold 0.5
```

---

## Требования к ресурсам

### Минимальные требования:

- **RAM:** 4-8 GB (рекомендуется 8+ GB)
- **CPU:** Современный процессор (Intel Core i5/i7 или AMD Ryzen 5/7)
- **Диск:** ~2 GB свободного места (для модели и зависимостей)

### Потребление ресурсов:

**При загрузке модели:**
- RAM: ~2-3 GB (модель загружается в память)
- CPU: Минимальная нагрузка

**Во время инференса (на одно изображение):**
- RAM: +500 MB - 1 GB (временные буферы)
- CPU: 50-100% одного ядра (может использовать несколько ядер)
- Время обработки: 5-30 секунд на изображение (зависит от размера и CPU)

**Примерные времена обработки:**
- Изображение 640x480: ~5-10 секунд
- Изображение 1920x1080: ~15-30 секунд
- Изображение 4K: ~30-60 секунд

### Оптимизация производительности:

1. **Уменьшите размер входного изображения:**
   ```python
   # В конфигурации можно установить:
   cfg.INPUT.MIN_SIZE_TEST = 640  # вместо 800
   cfg.INPUT.MAX_SIZE_TEST = 1333  # вместо 1333
   ```

2. **Используйте меньше потоков:**
   ```python
   import torch
   torch.set_num_threads(2)  # Ограничьте количество потоков
   ```

3. **Обрабатывайте изображения батчами** (если обрабатываете много изображений)

---

## Решение проблем

### Проблема: "CUDA out of memory" или ошибки CUDA

**Решение:** Убедитесь, что используете CPU версию:
```python
cfg.MODEL.DEVICE = "cpu"
```

### Проблема: Медленная работа на CPU

**Это нормально!** CPU инференс в 10-50 раз медленнее GPU. Рекомендации:
- Используйте меньшие изображения
- Обрабатывайте изображения ночью или в фоне
- Рассмотрите использование облачных GPU (Google Colab, AWS)

### Проблема: Ошибки компиляции при установке Detectron2

**Решение:**
1. Убедитесь, что установлены Visual Studio Build Tools
2. Установите правильную версию PyTorch (CPU)
3. Используйте WSL2 вместо нативного Windows

### Проблема: "ModuleNotFoundError: No module named 'detectron2'"

**Решение:**
```bash
# Проверьте установку
pip list | grep detectron2

# Переустановите
pip uninstall detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## Пример полного workflow

### 1. Копирование файлов с сервера

```powershell
# Создайте папку для проекта
mkdir detectron2_local
cd detectron2_local

# Скопируйте модель
scp -i ..\ssh\key root@91.236.199.226:/tmp/model_final.pth ./

# Скопируйте конфигурацию (или скачайте с GitHub)
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml" -OutFile "mask_rcnn_R_50_FPN_1x.yaml"
```

### 2. Установка в WSL2

```bash
# В WSL2
cd /mnt/c/path/to/detectron2_local
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
pip3 install opencv-python pillow
```

### 3. Запуск инференса

```bash
python3 inference_cpu.py \
  --image test_image.png \
  --weights model_final.pth \
  --config-file mask_rcnn_R_50_FPN_1x.yaml \
  --output-dir output \
  --num-classes 1 \
  --thing-classes frame
```

---

## Альтернативные варианты

### Вариант 1: Google Colab (Бесплатный GPU)

Если нужна быстрая работа, используйте Google Colab с бесплатным GPU:

1. Загрузите модель в Google Drive
2. Откройте Colab notebook
3. Подключите GPU runtime
4. Запустите инференс

### Вариант 2: Docker на Windows

Можно запустить Docker контейнер с Detectron2 на Windows:

```bash
docker run -it --rm \
  -v C:/path/to/model:/models \
  -v C:/path/to/images:/images \
  detectron2/detectron2:latest \
  python inference_cpu.py --weights /models/model_final.pth --image /images/test.png
```

---

## Заключение

Запуск модели на Windows с CPU возможен, но требует:
- Правильной установки зависимостей (лучше через WSL2)
- Терпения (CPU инференс медленнее GPU в 10-50 раз)
- Достаточного количества RAM (8+ GB рекомендуется)

**Рекомендация:** Используйте WSL2 для наиболее простой установки и работы.
