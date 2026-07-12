# Artifacts

В git добавлены только легкие и воспроизводимые артефакты:

- `submissions/submission_soup_dino75_quality20_siglip05.csv` — финальный сабмит, 4290 строк;
- `artifacts/report_dinov2.json` — небольшой отчет из архива `imgpref_dinov2_work.zip`;
- `notebooks/ivan_german_image_preference_solution.ipynb` — оформленный Kaggle notebook с сохраненными выводами ключевых ячеек.

Большой архив признаков не добавлен в git, потому что он весит около 374.7 MB и содержит `.npy/.npz` кэш:

- local file: `C:/Users/Иван/Downloads/imgpref_dinov2_work.zip`
- SHA256: `6ec3271f1291566d11a14393c3b1a4e812c5d32efa28700f520c201cd3d60e5f`
- содержимое: `work/cache/*.npy`, `preds.npz`, `report.json`

Для воспроизведения на Kaggle архив нужно загрузить как Kaggle Dataset или приложить к notebook input. При восстановлении он распаковывается в `/kaggle/working/work`.
