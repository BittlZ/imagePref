# Experiment log

Краткий журнал основных экспериментов. Public score указан по результатам отправок на Kaggle leaderboard.

| Этап | Идея | Public score | Вывод |
|---|---|---:|---|
| Baseline | ConvNeXtV2 + CLIP224, BT + LightGBM | 0.68886 | Базовое воспроизведение решения |
| SigLIP | Добавлен `vit_large_patch16_siglip_384.webli` | 0.69079 | Устойчивый прирост от нового vision-language сигнала |
| DINOv2 | Добавлен `vit_large_patch14_reg4_dinov2.lvd142m` | 0.69401 | Лучший одиночный прирост, основа финального сабмита |
| Quality head | Handcrafted признаки качества + CatBoost | 0.69373 | Отдельно слабее, но содержит небольшой независимый сигнал |
| EVA/CLIP | Проверка EVA/CLIP backbone | 0.69279 | Не вошел в финальное решение |
| Rank-soup 90/10 | `90% DINOv2 + 10% quality` | 0.69404 | Небольшой прирост над DINOv2 |
| Final rank-soup | `75% DINOv2 + 20% quality + 5% SigLIP` | **0.69417** | Лучший public score, команда поднялась на 2-е место |

Главный практический вывод: после добавления DINOv2 новые тяжелые backbone давали убывающую отдачу, а аккуратный rank-soup с quality-сигналом оказался самым эффективным коротким улучшением.
