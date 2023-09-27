import pandas as pd
import time
import datetime

# Считываем таблицу с признаками
features = pd.read_csv('features.csv', index_col='match_id')
print(features)

features_test = pd.read_csv('features_test.csv', index_col='match_id')
print(features_test)

target = features['radiant_win']



# Удаляем признаки, связанные с итогами матча
features_dif=features.columns.difference(features_test.columns.values.tolist()).tolist()
features.drop(features_dif, axis=1, inplace=True)
print(features)

# Подсчет пропусков
missing_values = features.count().sort_values(ascending=False)
missing_values = missing_values[missing_values != len(features)]

# Вывод признаков с пропусками и количество пропусков
print("Признаки с пропусками:")
for col, missing_count in missing_values.items():
    print(f"{col}: {missing_count}")

# Замена пропусков на нули
features.fillna(0, inplace=True)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
# Количество деревьев для тестирования
n_estimators_values = [10, 20, 30]
# Задаем генератор разбиений для кросс-валидации
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Словарь для хранения результатов
results = {}
for n_estimators in n_estimators_values:
    start_time = datetime.datetime.now()
    # Создаем классификатор градиентного бустинга
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    # Список для хранения метрик ROC AUC для каждого фолда
    roc_auc_scores = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred)
        roc_auc_scores.append(roc_auc)
    print(f"Количество деревьев: {n_estimators}, Средний ROC AUC: {roc_auc}")
    print('Time elapsed:', datetime.datetime.now() - start_time)
    # Среднее значение ROC AUC для всех фолдов
    mean_roc_auc = np.mean(roc_auc_scores)
    results[n_estimators] = mean_roc_auc

# Вывод результатов
#for n_estimators, roc_auc in results.items():
#    print(f"Количество деревьев: {n_estimators}, Средний ROC AUC: {roc_auc}")


