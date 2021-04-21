import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics

columns = [
    "dst_host_srv_serror_rate", "dst_host_serror_rate", "serror_rate",
    "srv_serror_rate", "count", "flag", "same_srv_rate", "dst_host_srv_count",
    "dst_host_diff_srv_rate"
]

file = pd.read_csv("train_9feats.csv")
# #linear regression
# print("##############################")
# print("Linear regression Model")
# lin_models_stat = {}
# for i in range(0, len(columns)):
#     print(f"\t** {columns[i]} against Malicious")
#     x = np.array(file[columns[i]].to_numpy()).reshape((-1, 1))
#     y = np.array(file['Malicious'].to_numpy())
#     model = LinearRegression().fit(x, y)
#     r_sq = model.score(x, y)
#     print(f"Coefficient of determination: {r_sq}")
#     print(f"Intercept: {model.intercept_}")
#     print(f"Gradient: {model.coef_}")
#     lin_models_stat[columns[i]] = {
#         'r_sq': r_sq,
#         'intercept': model.intercept_,
#         'gradient': model.coef_[0],
#         'model': model
#     }
# # print(lin_models_stat)

# multi linear regression
models_stat = {}
for i in range(0, len(columns)):
    for j in range(i + 1, len(columns)):
        x = np.array(
            pd.DataFrame(file, columns=[columns[i], columns[j]]).to_numpy())
        y = np.array(file['Malicious'].to_numpy())
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print(f"Coefficient of determination: {r_sq}")
        print(f"Intercept: {model.intercept_}")
        print(f"Gradient: {model.coef_}")
        models_stat[columns[i] + ';' + columns[j]] = {
            'r_sq': r_sq,
            'intercept': model.intercept_,
            'gradient': model.coef_[0],
            'model': model
        }
    print(models_stat)

print(pd.DataFrame(file, columns=columns[:1] + columns[1:2]).to_numpy)

test = pd.read_csv("test_9feats.csv")
actual_y = np.array(test['Malicious'].to_numpy())
accuracies = {}
score_to_feature = {}
scores = []
for feature, model in models_stat.items():
    f = feature.split(';')
    x = np.array(pd.DataFrame(test, columns=[f[0], f[-1]]).to_numpy())
    y_pred = model['model'].predict(x)
    y_pred = np.rint(y_pred)
    df = pd.DataFrame({'Actual': actual_y, 'Predicted': y_pred})
    # print(f"\t** {feature}")
    # print(df)
    accuracies[feature] = {
        'MAE': metrics.mean_absolute_error(actual_y, y_pred),
        'MSE': metrics.mean_squared_error(actual_y, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(actual_y, y_pred))
    }
    score_to_feature[accuracies[feature]['RMSE']] = feature
    scores.append(accuracies[feature]['RMSE'])

print(accuracies)
scores.sort()
for i in range(10):
    print(score_to_feature[scores[i]], scores[i])

# predict 1 feature linear regression
# print(pd.DataFrame(file, columns=columns[:1] + columns[1:2]).to_numpy)

# test = pd.read_csv("test_9feats.csv")
# actual_y = np.array(test['Malicious'].to_numpy())
# accuracies = {}
# score_to_feature = {}
# scores = []
# for feature, model in lin_models_stat.items():
#     y_pred = model['model'].predict(
#         np.array(test[feature].to_numpy()).reshape((-1, 1)))
#     y_pred = np.rint(y_pred)
#     df = pd.DataFrame({'Actual': actual_y, 'Predicted': y_pred})
#     # print(f"\t** {feature}")
#     # print(df)
#     accuracies[feature] = {
#         'MAE': metrics.mean_absolute_error(actual_y, y_pred),
#         'MSE': metrics.mean_squared_error(actual_y, y_pred),
#         'RMSE': np.sqrt(metrics.mean_squared_error(actual_y, y_pred))
#     }
#     score_to_feature[accuracies[feature]['RMSE']] = feature
#     scores.append(accuracies[feature]['RMSE'])

# print(accuracies)
# scores.sort()
# for i in range(5):
#     print(score_to_feature[scores[i]], scores[i])