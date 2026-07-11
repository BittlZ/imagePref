# Image Preference: решение соревнования

Репозиторий содержит воспроизводимое решение задачи выбора лучшего изображения из пары сгенерированных картинок.

- **Целевая переменная:** `is_image1_better`
- **Метрика:** ROC AUC
- **Лучший public score:** **0.69417**
- **Результат:** решение подняло команду на **2-е место** public leaderboard на момент отправки.

## Кратко о подходе

Решение построено как двухэтапный пайплайн:

1. **Frozen feature extraction.** Для каждого изображения извлекаются эмбеддинги из замороженных `timm` backbone. Признаки кэшируются в `.npy`, чтобы тяжелый GPU-этап не повторять при каждом эксперименте.
2. **Pairwise heads and rank blending.** На готовых эмбеддингах обучаются быстрые головы для сравнения пары: Bradley-Terry MLP и LightGBM. Финальное улучшение получено через rank-soup нескольких сабмитов.

Финальный ансамбль:

```text
75% DINOv2 submission
20% quality/artifact-head submission
 5% SigLIP submission
```

Ранговое смешивание выбрано потому, что ROC AUC зависит от порядка объектов, а не от абсолютной калибровки вероятностей.

## Использованные backbone

| Backbone | Назначение |
|---|---|
| `convnextv2_base.fcmae_ft_in22k_in1k_384` | детали, текстуры, артефакты генерации |
| `vit_large_patch14_clip_224.openai` | CLIP-семантика и preference-сигнал |
| `vit_large_patch16_siglip_384.webli` | дополнительный vision-language сигнал |
| `vit_large_patch14_reg4_dinov2.lvd142m` | self-supervised структура и композиция |

## Структура ветки

```text
image_preference/        # пакет пайплайна: config, features, models, submit
kaggle/experiments/      # экспорт ключевых Kaggle-ячеек в .py
notebooks/               # оформленный Kaggle notebook
submissions/             # финальный lightweight submission CSV
docs/                    # журнал экспериментов и описание внешних артефактов
artifacts/               # маленькие отчеты; большие кэши не коммитятся
```

## Основные файлы

- `notebooks/ivan_german_image_preference_solution.ipynb` — аккуратно оформленный notebook с русскими markdown-пояснениями и сохраненными выводами ключевых ячеек.
- `submissions/submission_soup_dino75_quality20_siglip05.csv` — лучший сабмит этой ветки.
- `docs/EXPERIMENTS.md` — таблица экспериментов и public score.
- `docs/ARTIFACTS.md` — описание большого архива признаков `imgpref_dinov2_work.zip`, который не добавлен в git.

## Запуск на Kaggle

1. Подключить dataset соревнования `teta-nn-2-2026`.
2. Подключить код решения или этот репозиторий как Kaggle Dataset.
3. Для быстрого воспроизведения подключить архив признаков `imgpref_dinov2_work.zip` и распаковать его в `/kaggle/working/work`.
4. Открыть `notebooks/ivan_german_image_preference_solution.ipynb` или последовательно выполнить скрипты из `kaggle/experiments/`.

Большой кэш `.npy/.npz` не хранится в git. Это сделано намеренно: репозиторий остается легким, а воспроизводимость обеспечивается через описание артефактов и финальный submission CSV.

## Локальная установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Полный feature extraction требует GPU и доступа к весам `timm`. Обучение голов из уже готовых `.npy` признаков можно выполнять значительно быстрее.
