"""
Скрипт для просмотра метрик обученной модели.

Пример использования:
    python train/view_metrics.py --output-dir output/my_target
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_metrics(metrics_file: Path) -> List[Dict]:
    """Загружает метрики из JSON файла."""
    metrics = []
    if not metrics_file.exists():
        print(f"Файл метрик не найден: {metrics_file}")
        return metrics
    
    with open(metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return metrics


def print_training_metrics(metrics: List[Dict]):
    """Выводит метрики обучения."""
    if not metrics:
        print("Метрики обучения не найдены")
        return
    
    # Находим последние метрики
    last_metrics = metrics[-1]
    
    print("\n" + "="*60)
    print("МЕТРИКИ ОБУЧЕНИЯ (последняя итерация)")
    print("="*60)
    
    if "iteration" in last_metrics:
        print(f"Итерация: {last_metrics['iteration']}")
    
    # Выводим основные метрики
    metric_keys = [
        "total_loss", "loss", "loss_classifier", "loss_box_reg", 
        "loss_mask", "loss_rpn_cls", "loss_rpn_box", "loss_objectness",
        "lr", "time", "data_time"
    ]
    
    for key in metric_keys:
        if key in last_metrics:
            value = last_metrics[key]
            if isinstance(value, float):
                print(f"{key:20s}: {value:.6f}")
            else:
                print(f"{key:20s}: {value}")
    
    # Статистика по всем итерациям
    if len(metrics) > 1:
        print("\n" + "-"*60)
        print("СТАТИСТИКА ПО ВСЕМУ ОБУЧЕНИЮ")
        print("-"*60)
        
        # Собираем все значения для каждой метрики
        metric_dict = {}
        for m in metrics:
            for key, value in m.items():
                if key != "iteration" and isinstance(value, (int, float)):
                    if key not in metric_dict:
                        metric_dict[key] = []
                    metric_dict[key].append(value)
        
        for key in ["total_loss", "loss", "loss_mask", "loss_classifier"]:
            if key in metric_dict:
                values = metric_dict[key]
                print(f"\n{key}:")
                print(f"  Начальное значение: {values[0]:.6f}")
                print(f"  Финальное значение: {values[-1]:.6f}")
                print(f"  Минимальное значение: {min(values):.6f}")
                print(f"  Максимальное значение: {max(values):.6f}")
                print(f"  Среднее значение: {sum(values)/len(values):.6f}")


def find_evaluation_results(output_dir: Path):
    """Ищет результаты оценки в директории."""
    eval_dirs = []
    
    # Ищем директории inference в output_dir
    for subdir in output_dir.iterdir():
        if subdir.is_dir() and "inference" in subdir.name.lower():
            eval_dirs.append(subdir)
    
    return eval_dirs


def print_evaluation_summary(output_dir: Path):
    """Выводит краткую сводку по оценке."""
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*60)
    
    # Ищем файлы с результатами оценки
    log_file = output_dir / "log.txt"
    if log_file.exists():
        print(f"\nЛог файл найден: {log_file}")
        print("Ищите строки с 'copypaste:' для метрик COCO")
        print("Пример: AP,AP50,AP75,APs,APm,APl")
        
        # Пытаемся найти последние метрики оценки в логе
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            found_eval = False
            for i, line in enumerate(lines[-100:]):  # Последние 100 строк
                if "copypaste:" in line and ("AP" in line or "segm" in line or "bbox" in line):
                    found_eval = True
                    print(f"\nНайдены метрики оценки:")
                    print(f"  {line.strip()}")
                    # Показываем следующую строку с числами
                    if i + 1 < len(lines):
                        next_line = lines[-100:][i + 1] if i + 1 < len(lines[-100:]) else ""
                        if next_line and ("copypaste:" in next_line or any(c.isdigit() for c in next_line)):
                            print(f"  {next_line.strip()}")
            
            if not found_eval:
                print("\nМетрики оценки не найдены в логе.")
                print("Запустите оценку с помощью:")
                print("  python train/train_single_class.py --eval-only --weights <model_path>")
    
    # Проверяем наличие директорий inference
    inference_dirs = find_evaluation_results(output_dir)
    if inference_dirs:
        print(f"\nНайдены директории с результатами оценки: {len(inference_dirs)}")
        for eval_dir in inference_dirs:
            print(f"  - {eval_dir}")


def export_to_csv(metrics: List[Dict], output_file: Path):
    """Экспортирует метрики в CSV файл."""
    if not metrics:
        print("Нет метрик для экспорта")
        return
    
    try:
        import pandas as pd
        df = pd.DataFrame(metrics)
        df.to_csv(output_file, index=False)
        print(f"\nМетрики экспортированы в CSV: {output_file}")
    except ImportError:
        print("Ошибка: pandas не установлен. Установите: pip install pandas")
        # Альтернативный способ без pandas
        import csv
        if metrics:
            keys = set()
            for m in metrics:
                keys.update(m.keys())
            keys = sorted(list(keys))
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(metrics)
            print(f"\nМетрики экспортированы в CSV (без pandas): {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Просмотр метрик обученной модели")
    parser.add_argument(
        "--output-dir", 
        required=True, 
        type=str, 
        help="Директория с обученной моделью (например, output/my_target)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Экспортировать метрики в CSV файл (опционально)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Ошибка: Директория не найдена: {output_dir}")
        return
    
    # Загружаем метрики обучения
    metrics_file = output_dir / "metrics.json"
    metrics = load_metrics(metrics_file)
    
    if metrics:
        print_training_metrics(metrics)
        
        # Экспорт в CSV если запрошен
        if args.export_csv:
            export_to_csv(metrics, Path(args.export_csv))
    else:
        print(f"Метрики обучения не найдены в {metrics_file}")
    
    # Показываем результаты оценки
    print_evaluation_summary(output_dir)
    
    # Показываем информацию о модели
    model_file = output_dir / "model_final.pth"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print("ИНФОРМАЦИЯ О МОДЕЛИ")
        print("="*60)
        print(f"Файл модели: {model_file}")
        print(f"Размер: {size_mb:.2f} MB")
    
    # Показываем конфигурацию если есть
    config_file = output_dir / "config.yaml"
    if config_file.exists():
        print(f"\nКонфигурация сохранена в: {config_file}")


if __name__ == "__main__":
    main()
