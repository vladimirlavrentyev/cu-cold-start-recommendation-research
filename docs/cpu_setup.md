# TaNP: запуск на CPU Mac M1

Это копия авторского репозитория TaNP с дополнительными скриптами для запуска на CPU. Исходная статья: *Task-adaptive Neural Process for User Cold-Start Recommendation*, WWW 2021.

## Что подготовлено

- Авторский commit: `994f2a2e305bbee8ce10cccbf158811600be49ff`.
- Данные из `TaNP/data.zip` уже распакованы.
- Архив содержит только подготовленный набор `lastfm_20`. MovieLens-1M в авторский архив не входит.
- `scripts/setup_cpu.sh` создаёт отдельное виртуальное окружение `.venv`.
- `run_experiment.py` является основной точкой запуска.
- Файлы в `scripts/legacy/` сохранены только для сопоставления со старым способом запуска.

## Первый запуск

Из корня репозитория выполните:

```bash
chmod +x scripts/setup_cpu.sh
./scripts/setup_cpu.sh
python run_experiment.py --config configs/smoke.json
```

Лог тестового запуска появится в:

```text
artifacts/runs/tanp_lastfm_smoke/tanp_lastfm_smoke.log
```

Если одна эпоха завершилась без ошибок, можно запустить полный эксперимент:

```bash
python run_experiment.py --config configs/baseline_cpu.json
```

Результаты полного запуска будут сохранены в:

```text
artifacts/runs/tanp_lastfm_cpu_baseline/
```

## Собственное число эпох

Скопируйте один из JSON-файлов в `configs/`, измените `run_name` и `num_epoch`, затем передайте новый файл в `run_experiment.py`.

## Важные замечания

1. Авторский репозиторий рассчитан на Python 3.6 и PyTorch 1.1. CPU-файлы используют PyTorch 2.2.2, доступный для Apple Silicon. Алгоритм и архитектура не изменены.
2. В коде тестирование выполняется после каждой эпохи, поэтому полный запуск может занимать заметно больше времени, чем одна только тренировка.
3. Сначала измерьте время одной эпохи через `run_cpu_smoke.sh`.
4. Файл `output_att` создаётся в `third_party/TaNP/` в конце запуска и содержит распределения по мягким кластерам.
5. Для строгой репликации результатов MovieLens-1M потребуется отдельно подготовить этот датасет по протоколу статьи.
