from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train model and return accuracy and F1 score.
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, f1