# Источники литературы

| ID | Citation | PDF file | Role in project | Use in paper | Notes |
|---|---|---|---|---|---|
| R1 | Pereira, Florbela. (2023). Machine Learning for the Prediction of Ionization Potential and Electron Affinity Energies Obtained by Density Functional Theory. *ChemistrySelect*, 8(5). | [PDF недоступен] | Центральная статья по теме; напрямую предсказывает IP и EA по DFT-данным; включает Random Forest, SVM, MLP и LightGBM. | Introduction, Methods, Related Work | ConnectTimeout при скачивании; возможно недоступен. |
| R2 | Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1, 140022. | [PDF недоступен] | Основной источник по QM9; нужен для описания датасета малых молекул и DFT-свойств. | Dataset, Introduction | HTTP 522 Server Error при скачивании; возможно временная недоступность. |
| R3 | Montavon, G., Rupp, M., Gobre, V., Vazquez-Mayagoitia, A., Hansen, K., Tkatchenko, A., Müller, K.-R., & von Lilienfeld, O. A. (2013). Machine Learning of Molecular Electronic Properties in Chemical Compound Space. *New Journal of Physics*, 15, 095003. | [R3_Montavon_2013_Molecular_Electronic_Properties.pdf](pdf/R3_Montavon_2013_Molecular_Electronic_Properties.pdf) (17 стр.) | Ранняя работа по ML для электронных свойств молекул; включает electron affinity как одну из предсказываемых электронных характеристик. | Introduction, Related Work, Methods | Полезна для обзора ранних подходов к ML-предсказанию EA. |
| R4 | Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*, 9, 513–530. | [R4_Wu_2017_MoleculeNet.pdf](pdf/R4_Wu_2017_MoleculeNet.pdf) (39 стр.) | Бенчмарк молекулярного ML; нужен для описания стандартных датасетов, метрик, фичей и baseline-подходов. | Dataset, Methods, Related Work | Содержит описание QM9 и других датасетов, метрик оценки. |
| R5 | Valdés, J. J., & Tchagang, A. B. (2024). Novel Machine Learning Insights into the QM7b and QM9 Quantum Mechanics Datasets. *Journal of Computational Chemistry*, 45(11), 791–804. | [PDF недоступен] | Анализ структуры QM7b и QM9; полезен для понимания свойств датасета, кластеров, сложности задачи и ограничений. | Dataset, Discussion | HTTP 403 Forbidden; paywall? |
| R6 | Ullah, A., Dral, P. O., & Tkatchenko, A. (2024). Molecular Quantum Chemical Data Sets and Databases for Machine Learning Potentials. *Chemical Reviews*, 124(11), 6727–6775. | [R6_Ullah_2024_QM_Datasets_Review.pdf](pdf/R6_Ullah_2024_QM_Datasets_Review.pdf) (62 стр.) | Обзор квантово-химических датасетов; нужен для контекста выбора QM9/QM7b и ограничений разных наборов. | Dataset, Related Work, Discussion | Свежий обзор (2024); полезен для обоснования выбора датасета. |
| R7 | Karandashev, K., Gasteiger, J., Margraf, J. T., & von Lilienfeld, O. A. (2022). An orbital-based representation for accurate quantum machine learning. *Journal of Chemical Physics*, 156(11), 114101. | [PDF недоступен] | Источник по представлениям молекул для ML электронных свойств; полезен для раздела о дескрипторах и representation choice. | Methods, Related Work | HTTP 403 Forbidden; paywall. |
| R8 | Fediai, A., Montavon, G., Müller, K.-R., & Tkatchenko, A. (2023). Accurate GW frontier orbital energies of 134 kilo molecules. *Scientific Data*, 10, 581. | [R8_Fediai_2023_GW_QM9.pdf](pdf/R8_Fediai_2023_GW_QM9.pdf) (20 стр.) | Уточнение электронных свойств QM9 на более высоком уровне теории; полезно для обсуждения различий между DFT-targets, orbital energies и физической интерпретацией EA. | Dataset, Discussion, Limitations | Содержит уточнённые значения орбитальных энергий; важно для понимания ограничений DFT-целей. |
| R9 | Zhan, Y., Ren, X., Zhao, S., & Guo, Z. (2025). Enhancing prediction of electron affinity and ionization energy in liquid organic electrolytes for lithium-ion batteries using machine learning. *Journal of Power Sources*, 612, 234–245. | [PDF не скачивался] | Свежая прикладная работа по EA/IE с линейной регрессией, Gradient Boosting, CatBoost, XGBoost и Random Forest; использовать как современный контекст. | Introduction, Related Work, Discussion | Landing page возвращает HTTP 403; скорее всего за paywall. Не скачивать. |

---

## Краткое описание ролей

- **R1, R3, R9** — ML для предсказания EA (центральные работы).
- **R2, R4, R5, R6, R8** — датасеты QM9/QM7b и их свойства.
- **R7** — представления молекул для ML.
- **R4, R6, R8** — также содержат обсуждение ограничений и точности.

## Примечания к скачиванию

- Успешно скачаны: R3, R4, R6, R8.
- Не скачаны: R1 (timeout), R2 (522 error), R5 (403), R7 (403), R9 (403/paywall).
- Подробный лог: [download_log.md](download_log.md).