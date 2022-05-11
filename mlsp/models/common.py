# #################################################################################################################### #
#       common.py                                                                                                      #
#           Common functions used by other functions in this folder.                                                   #
# #################################################################################################################### #

from colorama import Fore, Style
from sklearn import metrics


def process_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if verbose:
        print_confusion_matrix(y_test, prediction)
        print_classification_rprt(y_test, prediction)

    # Return the best model
    acc_score, train_score, test_score = print_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy_score": acc_score, "train_score": train_score, "test_score": test_score}


def print_confusion_matrix(y_test, prediction):
    print(f"Confusion matrix: {Fore.LIGHTGREEN_EX}{metrics.confusion_matrix(y_test, prediction)}")


def print_classification_rprt(y_test, prediction):
    print(f"Classification report: {Fore.LIGHTGREEN_EX}{metrics.classification_report(y_test, prediction)}")


def print_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    accuracy_score = round(metrics.accuracy_score(y_test, prediction), 2)
    model_train_score = round(model.score(x_train, y_train) * 100, 2)
    model_test_score = round(model.score(x_test, y_test) * 100, 2)

    print((
        f"Best achieved accuracy: {Fore.LIGHTGREEN_EX}{accuracy_score}"
        f"{Fore.WHITE}{Style.DIM} (train: {model_train_score}%"
        f", test: {model_test_score}%){Style.RESET_ALL}"
    ))

    return accuracy_score, model_train_score, model_test_score
