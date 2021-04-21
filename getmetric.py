def getmetrics(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn), tp / (tp + fn), tp / (tp + fp)


feat5 = getmetrics(62388, 48554, 10076, 4955)
print(feat5)