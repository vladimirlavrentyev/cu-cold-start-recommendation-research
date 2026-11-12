# Перенос группового знания в cold-start рекомендациях

Исследовательский проект курса «Научная студия». Я изучаю, как групповое знание в нейронных процессах влияет на рекомендации новым пользователям. Базовые модели — TaNP и FANP. Первая задача проекта — воспроизвести TaNP и получить полностью повторяемый эксперимент.

## Исследовательская гипотеза

Для отдельных новых пользователей групповое знание может ухудшать качество рекомендаций по сравнению с моделью, использующей только личную историю.

## Текущий этап

- авторский код TaNP сохранён отдельно в `third_party/TaNP`;
- авторский Gating-FiLM включён как явный вариант декодера;
- подготовлены CPU-конфигурации и последовательный запуск трёх датасетов на Colab T4;
- компактные эпизоды хранят идентификаторы вместо гигантских one-hot-тензоров;
- вычисления и построение графиков разделены на разные модули;
- `run_experiment.py` служит единственной точкой запуска;
- результаты сохраняются отдельно от checkpoints и исходного кода.

## Быстрый старт

Подготовить окружение:

```bash
./scripts/setup_cpu.sh
```

Если данные ещё не распакованы:

```bash
cd third_party/TaNP
unzip data.zip
cd ../..
```

Проверить команду без обучения:

```bash
python run_experiment.py --config configs/smoke.json --dry-run
```

Запустить короткий эксперимент. Одна команда выполнит обучение, сохранит метрики и построит графики:

```bash
python run_experiment.py --config configs/smoke.json
```

Полный CPU-запуск:

```bash
python run_experiment.py --config configs/baseline_cpu.json
```

## Обучение Gating-FiLM на Colab T4

Готовый ноутбук: `notebooks/tanp_gating_film_colab.ipynb`.

Он выполняет четыре шага:

1. проверяет доступность CUDA и T4;
2. подготавливает LastFM, MovieLens-1M и реконструированный Gowalla;
3. последовательно обучает отдельный TaNP Gating-FiLM на каждом датасете;
4. сохраняет три checkpoint и общую таблицу метрик.

Те же команды без ноутбука:

```bash
python scripts/prepare_datasets.py --dataset all --seed 2020
python run_experiment.py --suite configs/tanp_gating_film_colab.json --dry-run
python run_experiment.py --suite configs/tanp_gating_film_colab.json --skip-existing
```

`requirements-colab.txt` намеренно не устанавливает PyTorch: используется уже
установленная в Colab CUDA-сборка.

### Ограничение воспроизводимости данных

Авторы опубликовали готовые эпизоды только для LastFM. MovieLens-1M строится из
официального архива GroupLens. Для Gowalla точный авторский сплит и negative
sampling недоступны, поэтому `scripts/prepare_datasets.py` создаёт
детерминированную реконструкцию из данных NGCF и записывает её параметры в
`manifest.json`. Сравнение с таблицей статьи для Gowalla будет приближённым,
пока не получены исходные данные авторов.

## Результаты запуска

- `artifacts/runs/<run_name>/` — checkpoint, исходный лог и точная команда запуска;
- `results/<run_name>_metrics.csv` — компактная таблица метрик;
- `results/tanp_gating_film_seed2020_summary.csv` — финальные метрики трёх датасетов;
- `paper/figures/<run_name>_metrics.pdf` — векторный рисунок для статьи;
- `paper/figures/<run_name>_metrics.png` — рисунок для быстрой проверки.

## Структура

```text
.
├── configs/                  параметры экспериментов
├── src/cold_start_research/ вычисления и визуализация
├── third_party/TaNP/        авторская реализация TaNP
├── data/                     локальные исследовательские датасеты
├── artifacts/runs/          локальные сырые результаты и checkpoints
├── results/                 компактные таблицы
├── paper/                   текст статьи и итоговые рисунки
├── slides/                  материалы докладов
├── scripts/                 настройка окружения и служебные команды
├── docs/                    технические заметки
└── run_experiment.py        единая точка запуска
```

## Внешний код

TaNP: Xixun Lin et al., *Task-adaptive Neural Process for User Cold-Start Recommendation*, WWW 2021. Исходный репозиторий подключён как Git remote `upstream`.
