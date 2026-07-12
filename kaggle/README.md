# Kaggle experiments

Эта папка содержит кодовые фрагменты, вынесенные из оформленного ноутбука.
Скрипты рассчитаны на запуск внутри Kaggle Notebook с подключенными input dataset:

- данные соревнования `teta-nn-2-2026`;
- код решения `image_preference`;
- при необходимости архив признаков `imgpref_dinov2_work.zip`.

Рекомендуемый порядок воспроизведения финального сабмита:

1. `01_siglip_features.py`
2. `02_dinov2_features.py`
3. `03_quality_head.py`
4. `04_final_rank_soup.py`

`05_fast_stack_experimental.py` сохранен как исследовательская ветка: она не является финальным сабмитом.
