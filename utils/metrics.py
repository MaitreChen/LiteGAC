from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def classifier_evaluate(label, pred, average_mode='macro'):
    accuracy = accuracy_score(label, pred, normalize=True)

    precision = precision_score(label, pred, average=average_mode)
    recall = recall_score(label, pred, average=average_mode)
    f1 = f1_score(label, pred, average=average_mode)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)

    return accuracy, precision, recall, f1, cm


