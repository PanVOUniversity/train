# Гайд по запуску HTTP сервера для просмотра файлов

Этот гайд описывает различные способы запуска HTTP сервера для просмотра файлов из Docker контейнера на удаленном сервере.

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Вариант 1: HTTP сервер в контейнере](#вариант-1-http-сервер-в-контейнере)
3. [Вариант 2: HTTP сервер на хосте с SSH туннелем](#вариант-2-http-сервер-на-хосте-с-ssh-туннелем)
4. [Вариант 3: Просмотр нескольких папок](#вариант-3-просмотр-нескольких-папок)
5. [Альтернативные методы](#альтернативные-методы)

---

## Быстрый старт

**Самый простой способ:**

1. В Docker контейнере:
   ```bash
   cd ~/detectron2_repo
   python3 -m http.server 8000
   ```

2. На сервере (вне контейнера, как root):
   ```bash
   docker cp cda53d8848ba:/home/appuser/detectron2_repo/data /tmp/data
   docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/inference /tmp/inference
   mkdir -p /tmp/web_view
   ln -s /tmp/data /tmp/web_view/data
   ln -s /tmp/inference /tmp/web_view/inference
   cd /tmp/web_view
   python3 -m http.server 8000
   ```

3. На Windows компьютере:
   ```bash
   ssh -i .\ssh\key -L 8000:localhost:8000 root@91.236.199.226
   ```

4. В браузере:
   ```
   http://localhost:8000/data/
   http://localhost:8000/inference/
   ```

---

## Вариант 1: HTTP сервер в контейнере

### Шаг 1: Запуск сервера в контейнере

```bash
# В Docker контейнере
cd ~/detectron2_repo/output/inference
python3 -m http.server 8000
```

**Важно:** Не закрывайте терминал! Сервер работает только пока процесс запущен.

### Шаг 2: Копирование файлов из контейнера на хост

На сервере (вне контейнера, как root):

```bash
# Скопируйте папку из контейнера на хост
docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/inference /tmp/inference
```

### Шаг 3: Запуск HTTP сервера на хосте

```bash
# На сервере (вне контейнера)
cd /tmp/inference
python3 -m http.server 8000
```

### Шаг 4: Создание SSH туннеля

На вашем Windows компьютере:

```bash
ssh -i .\ssh\key -L 8000:localhost:8000 root@91.236.199.226
```

### Шаг 5: Просмотр файлов в браузере

Откройте в браузере:
```
http://localhost:8000/masks/page_9_combined_mask.png
http://localhost:8000/page_9_visualization.png
```

---

## Вариант 2: HTTP сервер на хосте с SSH туннелем

Этот вариант рекомендуется, если Docker контейнер запущен на удаленном сервере.

### Шаг 1: Копирование файлов из контейнера

На сервере (вне контейнера, как root):

```bash
# Скопируйте нужные папки из контейнера
docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/inference /tmp/inference
docker cp cda53d8848ba:/home/appuser/detectron2_repo/data /tmp/data
```

### Шаг 2: Запуск HTTP сервера на хосте

```bash
# Для одной папки
cd /tmp/inference
python3 -m http.server 8000

# Или для корневой структуры
mkdir -p /tmp/web_view
ln -s /tmp/data /tmp/web_view/data
ln -s /tmp/inference /tmp/web_view/inference
cd /tmp/web_view
python3 -m http.server 8000
```

### Шаг 3: Создание SSH туннеля

На Windows компьютере (PowerShell или CMD):

```bash
ssh -i .\ssh\key -L 8000:localhost:8000 root@91.236.199.226
```

**Важно:** Оставьте это окно открытым! Туннель работает только пока SSH сессия активна.

### Шаг 4: Просмотр в браузере

Откройте браузер и перейдите по адресу:
```
http://localhost:8000/
```

Вы увидите список файлов и папок, по которым можно навигировать.

---

## Вариант 3: Просмотр нескольких папок

### Способ A: Один сервер из корня проекта

**В Docker контейнере:**

```bash
cd ~/detectron2_repo
python3 -m http.server 8000
```

**На сервере (вне контейнера):**

```bash
# Скопируйте всю структуру
docker cp cda53d8848ba:/home/appuser/detectron2_repo /tmp/detectron2_repo

# Запустите сервер из корня
cd /tmp/detectron2_repo
python3 -m http.server 8000
```

**В браузере:**
- `http://localhost:8000/data/` - папка data
- `http://localhost:8000/output/inference/` - папка inference

### Способ B: Два сервера на разных портах

**В Docker контейнере (терминал 1):**

```bash
cd ~/detectron2_repo/data
python3 -m http.server 8000
```

**В Docker контейнере (терминал 2):**

```bash
cd ~/detectron2_repo/output/inference
python3 -m http.server 8001
```

**На сервере (вне контейнера):**

```bash
# Скопируйте обе папки
docker cp cda53d8848ba:/home/appuser/detectron2_repo/data /tmp/data
docker cp cda53d8848ba:/home/appuser/detectron2_repo/output/inference /tmp/inference

# Запустите два сервера в фоне
cd /tmp/data && nohup python3 -m http.server 8000 > /dev/null 2>&1 &
cd /tmp/inference && nohup python3 -m http.server 8001 > /dev/null 2>&1 &
```

**На Windows:**

```bash
# Создайте два туннеля
ssh -i .\ssh\key -L 8000:localhost:8000 -L 8001:localhost:8001 root@91.236.199.226
```

**В браузере:**
- `http://localhost:8000/` - папка data
- `http://localhost:8001/` - папка inference

---

## Альтернативные методы

### Метод 1: Использование VS Code Remote SSH

1. Подключитесь к серверу через VS Code Remote SSH
2. Откройте файл `output/inference/masks/page_9_combined_mask.png`
3. Кликните правой кнопкой → "Download..." или просто откройте файл

### Метод 2: Использование scp

На Windows компьютере:

```bash
# Скопируйте всю папку inference
scp -i .\ssh\key -r root@91.236.199.226:/tmp/inference ./

# Или только конкретный файл
scp -i .\ssh\key root@91.236.199.226:/tmp/inference/masks/page_9_combined_mask.png ./
```

### Метод 3: Использование SFTP клиента

Используйте FileZilla, WinSCP или другой SFTP клиент:

- **Host:** `91.236.199.226`
- **Username:** `root`
- **Protocol:** SFTP
- **Key file:** `.ssh/key`
- **Remote directory:** `/tmp/inference` или `/tmp/data`

---

## Полезные команды

### Проверка работы сервера

```bash
# Проверьте, что сервер запущен
curl http://localhost:8000/

# Или с сервера
curl http://localhost:8000/masks/page_9_combined_mask.png
```

### Остановка сервера

```bash
# Найдите процесс
ps aux | grep "http.server"

# Остановите процесс
kill <PID>

# Или если запущен в фоне
pkill -f "http.server"
```

### Просмотр логов

Если сервер запущен в фоне с перенаправлением вывода:

```bash
# Просмотр логов
tail -f /tmp/http_server.log
```

---

## Решение проблем

### Проблема: "Connection refused" при SSH туннеле

**Решение:**
1. Убедитесь, что HTTP сервер запущен на хосте (не в контейнере)
2. Проверьте, что порт 8000 не занят: `netstat -tuln | grep 8000`
3. Убедитесь, что SSH туннель создан правильно

### Проблема: Файлы не видны в браузере

**Решение:**
1. Проверьте права доступа: `chmod -R 755 /tmp/inference`
2. Убедитесь, что файлы скопированы: `ls -la /tmp/inference/masks/`
3. Проверьте путь в браузере (должен заканчиваться на `/` для папок)

### Проблема: Сервер останавливается при закрытии терминала

**Решение:**
Используйте `nohup` или `screen`/`tmux`:

```bash
# С nohup
nohup python3 -m http.server 8000 > /tmp/http_server.log 2>&1 &

# Или с screen
screen -S http_server
python3 -m http.server 8000
# Нажмите Ctrl+A затем D для отсоединения
```

---

## Безопасность

⚠️ **Важно:** HTTP сервер без аутентификации доступен всем, кто имеет доступ к порту!

**Рекомендации:**
1. Используйте SSH туннель вместо прямого доступа к порту
2. Останавливайте сервер после использования
3. Не используйте HTTP сервер на продакшн серверах
4. Используйте временные порты (8000-8999)

---

## Примеры использования

### Просмотр результатов инференса

```bash
# После запуска inference.py
cd ~/detectron2_repo/output/inference
python3 -m http.server 8000
```

Затем в браузере:
- `http://localhost:8000/masks/` - все маски
- `http://localhost:8000/page_9_visualization.png` - визуализация

### Просмотр данных датасета

```bash
cd ~/detectron2_repo/data
python3 -m http.server 8000
```

Затем в браузере:
- `http://localhost:8000/coco/val/` - валидационные изображения
- `http://localhost:8000/masks/` - маски

---

## Заключение

HTTP сервер - простой и удобный способ просмотра файлов с удаленного сервера. Выберите метод, который лучше всего подходит для вашей ситуации:

- **Для быстрого просмотра:** Вариант 1 (в контейнере) или Вариант 2 (на хосте)
- **Для просмотра нескольких папок:** Вариант 3A (один сервер из корня)
- **Для постоянного доступа:** Используйте scp или SFTP клиент
