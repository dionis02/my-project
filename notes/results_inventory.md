# Results inventory

## Main notebook

- **Path**: dopanti_paper/dopanti_small_mols_final_exps.ipynb
- **Purpose**: Сравнительная оценка классических ML-моделей для предсказания энергии сродства к электрону (EA) малых молекул с акцентом на поведение на наименее похожих молекулах.
- **Main dataset used**: `compact_data_FHI_575molecules.csv` (575 молекул, колонки: CAS, SMILES, InChi, InChiKey, EA, eV, BG, eV)
- **Target variable**: EA, eV (энергия сродства к электрону в эВ, **DFT‑calculated**, подтверждено автором)
- **Molecular representation**: 
  - Линейные дескрипторы RDKit (208 признаков)
  - Структурные дескрипторы (asphericity, planarity)
  - Энергия запрещённой зоны (BG) как дополнительный признак
  - Молекулярные фингерпринты Morgan (радиус 2, 2048 бит) для расчёта сходства **(окончательный критерий dissimilarity, подтверждён автором)**
- **Models**:
  1. LinearRegression (LinReg)
  2. DecisionTreeRegressor (DesTree) — оптимизированный через GridSearchCV
  3. RandomForestRegressor (RandFor) — оптимизированный через GridSearchCV
  4. GradientBoostingRegressor (XGB) — оптимизированный через GridSearchCV
  5. SVR (Support Vector Regression) с ядром RBF
- **Validation strategy**: 
  - Repeated K-Fold: 10 фолдов, 3 повтора (всего 30 сплитов)
  - Стратегия сохранена в файле `splits_rep3kfold10_list_full_base_FHI_chemData`
- **Metrics**:
  - Для всего validation fold: RMSE (mse_all), R² (r2_all)
  - Для top-20 наиболее непохожих молекул в validation: RMSE (mse_top20_dissimilar), R² (r2_top20_dissimilar)
  - Среднее и максимальное сходство валидационных молекул с тренировочными (mean_max_sim_val_to_train, max_max_sim_val_to_train)
  - Среднее и максимальное сходство top-20 dissimilar молекул с тренировочными (mean_max_sim_top20, max_max_sim_top20)
- **Top-20 dissimilar molecule analysis**:
  - Для каждого validation fold вычисляется максимальное сходство (Tanimoto по фингерпринтам Morgan) каждой валидационной молекулы с тренировочными
  - Выбираются 20 молекул с наименьшим сходством (наиболее непохожие)
  - Метрики считаются отдельно для этой подвыборки
- **Main output files**:
  - `models_cv_raw.csv` — сырые метрики по каждому фолду и модели
  - `models_cv_summary_full.csv` — полная сводка (mean, std, median, min, max)
  - `models_cv_summary_compact.csv` — компактная сводка (mean ± std)
  - `models_cv_ranking.csv` — ранжирование моделей по R² (all и top-20)
  - `models_cv_repeat_summary.csv` — средние метрики по повторам (3 повтора)
  - Сохранённые модели: `best_dtree_reg_full_base_FHI_chemData`, `best_rand_for_reg_full_base_FHI_chemData`, `best_boost_full_base_FHI_chemData`

## Detected result files

| File | What it contains | Models | Metrics | Split type | Top-20 dissimilar? | Can be used in paper? | Notes |
|---|---|---|---|---|---|---|---|
| `compact_data_FHI_575molecules.csv` | Исходные данные: 575 молекул, SMILES, EA (эВ), BG (эВ) | — | — | — | — | Да (описание датасета) | 576 строк (заголовок + 575 молекул); разделитель табуляция |
| `models_cv_raw.csv` | Сырые метрики по каждому фолду (30 фолдов × 5 моделей = 150 строк) | LinReg, DesTree, RandFor, SVR, XGB | mse_all, r2_all, mse_top20_dissimilar, r2_top20_dissimilar, mean_max_sim_val_to_train, max_max_sim_val_to_train, mean_max_sim_top20, max_max_sim_top20 | Repeated K-Fold (10×3) | Да (отдельные колонки) | Да (детализация) | Столбцы: fold, mse_all, r2_all, mse_top20_dissimilar, r2_top20_dissimilar, mean_max_sim_val_to_train, max_max_sim_val_to_train, mean_max_sim_top20, max_max_sim_top20, model, repeat, fold_within_repeat |
| `models_cv_summary_full.csv` | Полная сводка: mean, std, median, min, max по каждой метрике и модели | LinReg, DesTree, RandFor, SVR, XGB | r2_all, r2_top20_dissimilar, mean_max_sim_val_to_train, mean_max_sim_top20 (с агрегатами) | — | Да | Да (основные результаты) | Двухуровневый заголовок; содержит статистики распределения |
| `models_cv_summary_compact.csv` | Компактная сводка: mean ± std для удобства чтения | LinReg, DesTree, RandFor, SVR, XGB | r2_all, r2_top20_dissimilar, mean_max_sim_val_to_train, mean_max_sim_top20 | — | Да | Да (ключевая таблица) | Формат: "0.637 ± 0.111" |
| `models_cv_ranking.csv` | Ранжирование моделей по R² (сортировка по r2_top20_dissimilar убыванию, затем r2_all убыванию) | XGB, LinReg, SVR, RandFor, DesTree | r2_top20_dissimilar, r2_all | — | Да | Да (ранжирование) | XGB лучший по обоим метрикам |
| `models_cv_repeat_summary.csv` | Средние метрики по каждому повтору (3 повтора) | LinReg, DesTree, RandFor, SVR, XGB | r2_all, r2_top20_dissimilar | По повторам | Да | Да (анализ стабильности) | Показывает воспроизводимость между повторами |
| `splits_rep3kfold10_list_full_base_FHI_chemData` | Сохранённые индексы train/val сплитов (30 сплитов) | — | — | Repeated K-Fold | — | Нет (технический файл) | binary joblib; используется для воспроизводимости |
| `best_dtree_reg_full_base_FHI_chemData` | Оптимизированная модель Decision Tree | DesTree | — | — | — | Нет (модель) | binary joblib; можно загрузить для предсказаний |
| `best_rand_for_reg_full_base_FHI_chemData` | Оптимизированная модель Random Forest | RandFor | — | — | — | Нет (модель) | binary joblib |
| `best_boost_full_base_FHI_chemData` | Оптимизированная модель Gradient Boosting | XGB | — | — | — | Нет (модель) | binary joblib |

## Important code blocks from notebook

1. **Data loading**
   - Загрузка CSV `compact_data_FHI_575molecules.csv` (разделитель табуляция)
   - Загрузка 3D структур из SDF `3d_charged_compact_data_FHI_575molecules.sdf`
   - Целевая переменная: колонка `EA, eV` (запятая как десятичный разделитель)

2. **Feature preparation**
   - Вычисление линейных дескрипторов RDKit через `MoleculeDescriptors` (208 признаков)
   - Добавление структурных дескрипторов: asphericity (степень несферичности) и planarity (планарность на основе двугранных углов)
   - Добавление энергии запрещённой зоны (`BG, eV`) как дополнительного признака
   - Масштабирование признаков через `StandardScaler`
   - Генерация фингерпринтов Morgan (радиус 2, 2048 бит) для расчёта сходства

3. **Model definitions**
   - LinearRegression (без параметров)
   - DecisionTreeRegressor — оптимизация через GridSearchCV по max_depth, min_samples_split, min_samples_leaf
   - RandomForestRegressor — оптимизация через GridSearchCV по n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
   - GradientBoostingRegressor — оптимизация через GridSearchCV по learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, subsample, max_leaf_nodes, loss, criterion
   - SVR с ядром RBF (по умолчанию)

4. **Cross-validation loop**
   - `RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)` — 30 сплитов
   - Функция `evaluate_model_with_novel_subset()` для оценки с top‑20 dissimilar
   - Для каждого фолда: обучение на train, предсказание на val, вычисление метрик на всём val и на top‑20 dissimilar

5. **Dissimilarity / top-20 molecule selection**
   - Функция `max_similarity_to_train()`: вычисляет максимальное сходство Tanimoto между каждой валидационной молекулой и тренировочными
   - Функция `select_most_dissimilar_val()`: сортирует по возрастанию сходства, выбирает top‑k (по умолчанию 20) наименее похожих

6. **Metric calculation**
   - RMSE (Root Mean Squared Error) и R² для двух подвыборок: all validation, top‑20 dissimilar
   - Среднее и максимальное сходство для обеих подвыборок
   - Все метрики сохраняются по каждому фолду

7. **Result saving**
   - Функция `summarize_cv_results()` генерирует 4 CSV файла: raw, summary_full, summary_compact, ranking, repeat_summary
   - Автоматическое добавление колонок repeat и fold_within_repeat для анализа по повторам

8. **Plotting / tables**
   - Boxplots распределения метрик по фолдам для каждой модели
   - Barplots mean ± std
   - Scatter plots: R² all vs R² top‑20 dissimilar, RMSE all vs RMSE top‑20 dissimilar
   - Scatter similarity vs performance
   - Все графики отображаются автоматически при вызове функции с `show_plots=True`

## Other notebooks in dopanti_paper/

- `dopanti_new_dataset.ipynb` — большой ноутбук (27 МБ), возможно, содержит дополнительные эксперименты или обработку данных.
- `Mordred_75_kin_retest_3.ipynb` — возможно, эксперименты с дескрипторами Mordred.
- `sol_1DCNN_updated.ipynb`, `sol_2DGNN_updated.ipynb` — нейросетевые модели (1D CNN, 2D GNN), выходят за границы проекта (классические ML).
- `sol_CATBOOST_upd_(bigger_morgan).ipynb` — возможно, эксперименты с CatBoost и расширенными фингерпринтами Morgan.

**Примечание:** Для целей данного проекта (классические ML) фокус на `dopanti_small_mols_final_exps.ipynb` достаточен, так как он содержит систематическое сравнение пяти классических моделей с валидацией и анализом dissimilar молекул. Остальные ноутбуки могут рассматриваться как дополнительные материалы, но не являются основным источником результатов для статьи.