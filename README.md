# image_preference

Модель для соревнования «какая из двух сгенерированных картинок лучше» (метрика — **ROC AUC**).

Подход: замороженные сильные бэкбоны (`timm`) → эмбеддинг на картинку → две головы поверх
(**Bradley–Terry** MLP и **LightGBM** на сравнительных фичах) → блендинг по OOF.
Опционально — end-to-end сиамский файнтюн как вариант с более высоким потолком.

Подробное описание — в [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Структура

Корень рабочей директории:

```
nn_2/
├── data/                     # сырые данные соревнования (локально)
│   ├── train.parquet
│   ├── test.parquet
│   └── sample_submission.csv
├── eda/                      # разведочный анализ (не входит в Kaggle-зип)
│   ├── eda.py
│   ├── overview.png
│   └── example_pairs.png
└── image_preference/         # деплой-проект (его и зипуем на Kaggle)
```

Сам проект:

```
image_preference/
├── imgpref/
│   ├── config.py         # Config + auto_config + загрузка YAML/JSON
│   ├── data.py           # чтение parquet, декод, Dataset'ы, метаданные
│   ├── features.py       # извлечение фич timm-бэкбонами + TTA + кеш
│   ├── pair_features.py  # сравнительные фичи пары (diff/prod/cos/meta)
│   ├── cv.py             # StratifiedKFold, ранк-усреднение, поиск весов блендинга
│   ├── submit.py         # запись submission.csv
│   ├── pipeline.py       # оркестрация: features → train → predict
│   ├── utils.py          # seed, device, ROC AUC без sklearn
│   └── models/
│       ├── bt_head.py    # Bradley–Terry голова (torch)
│       ├── gbdt.py       # LightGBM
│       └── finetune.py   # опциональный end-to-end сиамский файнтюн
├── configs/
│   └── default.yaml      # ВСЕ настройки пайплайна в одном файле
├── notebooks/kaggle_run.py   # шаблон ячеек Kaggle-ноутбука
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

## Запуск на Kaggle

1. **Зип кода (без данных!).** Из папки `image_preference/`:
   ```bash
   zip -r imgpref-code.zip imgpref configs notebooks requirements.txt README.md ARCHITECTURE.md
   ```
2. **Создать Kaggle Dataset** из `imgpref-code.zip` (например, имя `imgpref-code`).
3. **Новый Notebook**: приаттачить (a) данные соревнования, (b) датасет `imgpref-code`.
   Accelerator → **GPU T4 x2**. Internet → **ON** (чтобы качались веса `timm`).
4. Скопировать ячейки из `notebooks/kaggle_run.py` и выполнить по порядку:
   `run_features` → `run_train` → `run_predict`. Итог — `/kaggle/working/submission.csv`.

### Если интернет на Kaggle выключен
Скачать веса бэкбонов локально, залить отдельным датасетом и указать путь:
```python
cfg.backbones = [BackboneSpec("convnextv2_base.fcmae_ft_in22k_in1k_384",
                              img_size=384, weight_path="/kaggle/input/weights/convnextv2.pth")]
```

## Локальный прогон (для отладки)
```bash
pip install -r requirements.txt
python -c "from imgpref.config import auto_config; from imgpref.pipeline import run_all; run_all(auto_config())"
```

## Настройки — `configs/default.yaml`
Все гиперпараметры вынесены в один YAML-файл; `auto_config()` его читает и сам
подставляет пути (`data/` локально, `/kaggle/input` на Kaggle). Переопределить
на лету можно через kwargs: `auto_config(finetune_enabled=True, n_folds=5)`.

Ключевые ручки:
- `backbones` — список бэкбонов для ансамбля (имена `timm`).
- `use_tta_hflip` — TTA горизонтальным отражением при извлечении фич.
- `use_gbdt`, `gbdt_pair_mode` — LightGBM и режим сравнительных фич (`compact`/`full`).
- `finetune_enabled` — включить end-to-end файнтюн (дольше, выше потолок).
- `n_folds`, `seed`, `blend_search` — кросс-валидация и блендинг.
