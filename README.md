# Image Preference — соревнование по предсказанию лучшей картинки

Решение для задачи **«какая из двух сгенерированных картинок лучше»** (метрика — **ROC AUC**).

Подробная архитектура — в [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Задача

На входе пара изображений `(image_1, image_2)` — две генерации (часто по одному промпту, но **без текста промпта** в данных). Нужно предсказать вероятность того, что **`image_1` признана лучше** экспертной комиссией (`is_image1_better`).

| | train | test |
|---|---|---|
| Пар | 8710 | 4290 |
| Разрешение | 1000×1000 RGB | 1000×1000 RGB |
| Формат `image_1` | JPEG / PNG / WEBP | то же |
| Формат `image_2` | **всегда WEBP** | **всегда WEBP** |
| Баланс метки | `P(image_1 лучше) ≈ 0.35` | — |

По сути это **reward-модель человеческого предпочтения** между двумя картинками: важен не абсолютный скор «красоты», а **ранжирование** внутри пары (именно это оптимизирует ROC AUC).

---

## Наш подход

Двухэтапный пайплайн: тяжёлое извлечение признаков один раз на GPU, обучение голов — быстро на CPU из готовых эмбеддингов.

```
parquet (байты картинок)
        │
        ▼  [GPU] frozen timm-бэкбоны + hflip TTA → L2-норм эмбеддинг f на картинку
        │  сохраняем в .npy (по колонке: image_1 / image_2, train / test)
        ▼
   ┌────────────────────────────────────────────────────────────┐
   │  [CPU] для каждого бэкбона — своя Bradley–Terry MLP-голова │
   │        logit = s(f₁) − s(f₂)  (антисимметричный ранкинг)   │
   │  [CPU] LightGBM на сравнительных фичах пары + метаданные   │
   │        [f₁−f₂, f₁·f₂, |f₁−f₂|, cos, формат/размер байтов]  │
   └────────────────────────────────────────────────────────────┘
        │
        ▼  блендинг предсказаний по OOF (ранговая шкала) → submission.csv
```

**Ключевые решения:**

1. **Замороженные бэкбоны** (`timm`) — устойчиво на маленьком train (8710 пар); не переобучаем ViT с нуля.
2. **Ансамбль разных «взглядов»** — CNN (ConvNeXt-V2 @384) + self-supervised (DINOv2) + CLIP (семантика/«вкус», как в PickScore/HPS).
3. **Bradley–Terry** — общий скор `s(f)` на картинку, логит пары = разность; swap-аугментация и multi-seed снижают дисперсию.
4. **Блендинг отдельных голов**, а не слепая конкатенация всех эмбеддингов — слабые бэкбоны получают малый вес, сильные (CLIP) — большой.
5. **Метаданные** (формат `image_1`, log-размеры байтов) — дешёвый ортогональный сигнал для GBDT.
6. **Извлечение по одной колонке за GPU-сессию** (image_1 → restart → image_2) — обход OOM на Kaggle при parquet с одним row-group.

**Использованные бэкбоны (извлечение фич):**

| Бэкбон | Разрешение | Роль |
|---|---|---|
| `convnextv2_base` | 384 | текстура, деталь, артефакты генерации |
| `vit_large_patch14_dinov2` | 224 | структура, геометрия |
| `vit_large_patch14_clip_224` | 224 | семантика, preference-сигнал (сильнейший одиночный) |

---

## Предвычисленные признаки (features)

Готовые эмбеддинги для train/test (обе колонки, три бэкбона) лежат на Google Drive:

**[image_pref_data — Google Drive](https://drive.google.com/drive/folders/1JZI_STi_bKpe7IO-G9DIRQQjPZoegwIQ?usp=sharing)**

Файлы в `.npy` (рекомендуется) и `.csv` (опционально):

```
feat_train_image_1_<backbone>.npy   # (8710, 1024)
feat_train_image_2_<backbone>.npy
feat_test_image_1_<backbone>.npy    # (4290, 1024)
feat_test_image_2_<backbone>.npy
```

`<backbone>` ∈ `convnextv2_base`, `vit_large_patch14_dinov2`, `vit_large_patch14_clip_224`.

Скачай `.npy` в `data/` локально или залей как **Kaggle Dataset** — дальше обучение голов не требует GPU и декода картинок.

---

## Структура репозитория

```
nn_2/
├── data/                      # сырые данные + предвычисленные .npy (локально)
│   ├── train.parquet
│   ├── test.parquet
│   ├── sample_submission.csv
│   └── feat_*_<backbone>.npy
├── eda/                       # разведочный анализ (графики, eda.py)
├── image_preference/          # код пайплайна (зипуется на Kaggle)
│   ├── imgpref/               # пакет: config, features, models, pipeline, …
│   ├── configs/default.yaml   # гиперпараметры
│   └── notebooks/kaggle_run.py
├── ARCHITECTURE.md
└── README.md
```

---

## Быстрый старт на Kaggle (обучение из готовых фич)

1. **Dataset с кодом** — зип из `image_preference/` (см. [`image_preference/README.md`](image_preference/README.md)).
2. **Dataset с фичами** — `.npy` с [Google Drive](https://drive.google.com/drive/folders/1JZI_STi_bKpe7IO-G9DIRQQjPZoegwIQ?usp=sharing) или Kaggle Dataset.
3. **Notebook (CPU)** — инпуты: данные соревнования + код + фичи.

```python
# загрузка и конкатенация (пример — один бэкбон)
import glob, numpy as np
def find(name): return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]

bb = "vit_large_patch14_clip_224"
F1_tr = np.load(find(f"feat_train_image_1_{bb}.npy"))
F2_tr = np.load(find(f"feat_train_image_2_{bb}.npy"))
# … далее train_bt_cv + train_lgbm_cv + блендинг (см. ноутбук в истории / ARCHITECTURE.md)
```

4. **Submit** — `/kaggle/working/submission.csv`.

---

## Извлечение новых фич (GPU)

Если нужны другие бэкбоны (CLIP@336, SigLIP) — GPU T4×2, Internet ON, **одна колонка за прогон** с restart kernel между `image_1` и `image_2`. Шаблон ячеек — в переписке / `notebooks/kaggle_run.py`; экспорт в `/kaggle/working/export/*.npy`.

---

## Настройки

Все гиперпараметры — в [`image_preference/configs/default.yaml`](image_preference/configs/default.yaml):

- `backbones` — список timm-моделей для извлечения
- `n_folds`, `seed`, `blend_search` — CV и блендинг
- `bt_*` — Bradley–Terry голова (hidden, lr, epochs, swap_aug)
- `lgbm_params`, `gbdt_pair_mode` — LightGBM
- `finetune_enabled` — опциональный end-to-end файнтюн (высокий потолок, дорого по GPU)

---

## Локально

```bash
cd image_preference
pip install -r requirements.txt
# полный пайплайн (нужны parquet в ../data/ и GPU для features):
python -c "from imgpref.config import auto_config; from imgpref.pipeline import run_all; run_all(auto_config())"
```

Для обучения только из `.npy` в `data/` достаточно CPU + `lightgbm` + `torch`.
