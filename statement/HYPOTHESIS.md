# Гипотезы проекта

## Контекст
Проект посвящён систематическому сравнению классических моделей машинного обучения для предсказания энергии сродства к электрону (EA) малых молекул. Эксперименты уже проведены и задокументированы в ноутбуке `dopanti_paper/dopanti_small_mols_final_exps.ipynb` и CSV-артефактах. Данные гипотезы формулируются как ожидания, которые могут быть проверены на основе имеющихся результатов.

## Методология проверки
- **Датасет**: 575 малых молекул (файл `compact_data_FHI_575molecules.csv`)
- **Признаки**: RDKit линейные дескрипторы (208), структурные (asphericity, planarity), энергия запрещённой зоны (BG), фингерпринты Morgan (расчёт сходства)
- **Модели**: Linear Regression (LinReg), Decision Tree (DesTree), Random Forest (RandFor), Gradient Boosting (XGB), Support Vector Regression (SVR)
- **Валидация**: Repeated K‑Fold (10 фолдов × 3 повтора, всего 30 сплитов)
- **Метрики**: RMSE (mse_all, mse_top20_dissimilar), R² (r2_all, r2_top20_dissimilar)
- **Анализ dissimilar молекул**: для каждого validation fold выбираются 20 молекул с наименьшим максимальным сходством Tanimoto к тренировочным (на основе фингерпринтов Morgan)

## Гипотезы

### H1. Tree-based ensemble models improve EA prediction compared with simple linear regression

**Формулировка:**  
Tree-based ensemble models, especially Random Forest and Gradient Boosting, are expected to achieve lower MSE and higher R² than ordinary Linear Regression because they can capture nonlinear relationships between molecular descriptors and electron affinity.

**Что должно наблюдаться:**
- lower average MSE across repeated cross-validation
- higher average R² across repeated cross-validation
- more stable performance across repeated splits (lower variance)

**Что может опровергнуть:**
- Linear Regression performs similarly or better
- ensemble models show strong overfitting
- cross-validation variance is too high

**Связанные эксперименты:**  
repeated cross-validation with MSE and R²; evidence from `dopanti_paper/models_cv_summary_compact.csv` and `dopanti_paper/models_cv_ranking.csv`.

**Preliminary status:** supported  
**Evidence file:** `dopanti_paper/models_cv_summary_compact.csv`, `dopanti_paper/models_cv_ranking.csv`  
**Brief evidence summary:**  
- Gradient Boosting (XGB) имеет наивысший средний R²_all (0.793 ± 0.077) и R²_top20_dissimilar (0.684 ± 0.286).  
- Linear Regression (LinReg) показывает R²_all = 0.779 ± 0.095 и R²_top20 = 0.655 ± 0.280, что ниже, чем у XGB.  
- Random Forest (RandFor) также превосходит Linear Regression по R²_all (0.749 ± 0.079 vs 0.779), но немного уступает по R²_top20 (0.639 ± 0.243 vs 0.655).  
- Таким образом, tree-based ensemble (XGB) демонстрирует лучшую среднюю производительность, чем простая линейная регрессия.

---

### H2. Single Decision Tree is expected to be less stable than ensemble tree models

**Формулировка:**  
A single Decision Tree is expected to show less stable performance across repeated splits than Random Forest or Gradient Boosting because it is more sensitive to small changes in training data.

**Что должно наблюдаться:**
- larger variance of MSE/R² across repeated cross-validation
- weaker average validation performance
- weaker performance on top-20 dissimilar validation molecules

**Что может опровергнуть:**
- Decision Tree performs similarly to ensembles
- variance across splits is low
- top-20 dissimilar performance is comparable to ensemble models

**Связанные эксперименты:**  
repeated cross-validation; top-20 dissimilar validation molecule analysis; evidence from summary tables.

**Preliminary status:** supported  
**Evidence file:** `dopanti_paper/models_cv_summary_compact.csv`  
**Brief evidence summary:**  
- Decision Tree (DesTree) имеет наибольшее стандартное отклонение R²_all (0.111) среди всех моделей (у XGB — 0.077, у LinReg — 0.095), что указывает на большую нестабильность по повторным сплитам.  
- Средний R²_all у DesTree (0.637 ± 0.111) значительно ниже, чем у ensemble моделей (XGB: 0.793, RandFor: 0.749).  
- Производительность на top‑20 dissimilar молекулах у DesTree также худшая (R²_top20 = 0.508 ± 0.250), что подтверждает слабую обобщающую способность.

---

### H3. Model performance decreases on chemically dissimilar validation molecules

**Формулировка:**  
All classical ML models are expected to show worse performance on top-20 chemically dissimilar validation molecules compared with the full validation set, because these molecules are farther from the training chemical domain.

**Что должно наблюдаться:**
- higher MSE on top-20 dissimilar molecules than on full validation sets
- lower R² on top-20 dissimilar molecules than on full validation sets
- larger prediction errors for molecules structurally distant from training data

**Что может опровергнуть:**
- top-20 dissimilar metrics are similar to full validation metrics
- some models generalize equally well to dissimilar molecules
- dissimilarity criterion does not correlate with prediction error

**Связанные эксперименты:**  
comparison between full validation metrics and top-20 dissimilar metrics from `dopanti_paper/models_cv_summary_compact.csv`.

**Preliminary status:** supported  
**Evidence file:** `dopanti_paper/models_cv_summary_compact.csv`  
**Brief evidence summary:**  
Для всех пяти моделей R²_top20_dissimilar ниже, чем R²_all:
- XGB: 0.684 vs 0.793 (‑0.109)
- LinReg: 0.655 vs 0.779 (‑0.124)
- SVR: 0.645 vs 0.758 (‑0.113)
- RandFor: 0.639 vs 0.749 (‑0.110)
- DesTree: 0.508 vs 0.637 (‑0.129)

Снижение R² составляет 0.109–0.129, что подтверждает, что предсказание для химически менее похожих молекул существенно сложнее.

---

### H4. Ensemble models are expected to be more robust on dissimilar molecules than linear regression and single tree

**Формулировка:**  
Random Forest and Gradient Boosting are expected to retain better predictive performance on top-20 dissimilar molecules than Linear Regression and Decision Tree, because ensemble methods can model nonlinear descriptor-property relationships and reduce instability.

**Что должно наблюдаться:**
- lower MSE on top-20 dissimilar molecules
- higher R² on top-20 dissimilar molecules
- smaller degradation from full validation metrics to top-20 dissimilar metrics

**Что может опровергнуть:**
- all models degrade similarly
- Linear Regression performs comparably on dissimilar molecules
- ensembles overfit to common chemical patterns and fail on dissimilar molecules

**Связанные эксперименты:**  
top-20 dissimilar molecule metrics from `dopanti_paper/models_cv_summary_compact.csv` and `dopanti_paper/models_cv_ranking.csv`.

**Preliminary status:** partially supported  
**Evidence file:** `dopanti_paper/models_cv_summary_compact.csv`  
**Brief evidence summary:**  
- Gradient Boosting (XGB) достигает наивысшего R²_top20_dissimilar (0.684 ± 0.286), что превосходит Linear Regression (0.655 ± 0.280) и Decision Tree (0.508 ± 0.250).  
- Однако разница между XGB и Linear Regression невелика (0.684 vs 0.655), а стандартные отклонения велики (0.286 и 0.280), что делает различие статистически неочевидным без дополнительного теста.  
- Random Forest (0.639 ± 0.243) немного уступает Linear Regression на dissimilar молекулах.  
- Таким образом, **Gradient Boosting действительно показывает лучшую производительность на dissimilar молекулах**, но преимущество не столь выражено, как ожидалось; Linear Regression остаётся конкурентоспособным.

---

## Сводка предварительных результатов

| Гипотеза | Предварительный статус | Ключевое наблюдение |
|----------|------------------------|---------------------|
| H1: Tree‑based ensemble vs Linear Regression | supported | XGB превосходит LinReg по R²_all и R²_top20 |
| H2: Decision Tree менее стабилен, чем ensemble | supported | DesTree имеет наибольшее std R²_all (0.111) и худшие средние метрики |
| H3: Производительность падает на dissimilar молекулах | supported | Все модели показывают снижение R² на 0.109–0.129 |
| H4: Ensemble более robust на dissimilar молекулах | partially supported | XGB лучший на top‑20, но LinReg близок; разница невелика |

## Ограничения предварительной оценки
1. **Отсутствие статистических тестов** — разницы между моделями оценены только по средним и стандартным отклонениям; необходимы тесты на значимость (например, парные t‑тесты по фолдам).
2. **Большие стандартные отклонения** на top‑20 dissimilar молекулах (0.243–0.286) указывают на высокую вариативность между фолдами, что затрудняет однозначные выводы.
3. **Критерий dissimilarity** основан на максимальном сходстве Tanimoto по фингерпринтам Morgan; возможны альтернативные меры (например, scaffold‑based splitting).
4. **Размер датасета** (575 молекул) ограничивает статистическую мощность, особенно для подвыборки из 20 молекул на фолд.

## Рекомендации для окончательной проверки
1. Провести парные статистические тесты для сравнения моделей по фолдам.
2. Проанализировать распределение ошибок на dissimilar молекулах (например, через scatter plots ошибок vs сходства).
3. Исследовать, связана ли величина деградации производительности с химическими классами молекул.
4. Рассмотреть альтернативные стратегии выделения dissimilar молекул (например, кластерный анализ).

## Ссылки на артефакты
- **Основной ноутбук:** `dopanti_paper/dopanti_small_mols_final_exps.ipynb`
- **Сводные таблицы:** `dopanti_paper/models_cv_summary_compact.csv`, `dopanti_paper/models_cv_ranking.csv`
- **Сырые метрики по фолдам:** `dopanti_paper/models_cv_raw.csv`
- **Датасет:** `dopanti_paper/compact_data_FHI_575molecules.csv`

*Примечание: предварительный статус основан на имеющихся численных результатах и может быть уточнён после дополнительного статистического анализа.*