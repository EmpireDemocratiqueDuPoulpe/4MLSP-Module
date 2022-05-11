# #################################################################################################################### #
#       svm.py                                                                                                         #
#           Models based on support vector machines algorithms.                                                        #
# #################################################################################################################### #

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from . import common


def linear_svc_model(preprocessor, x_train, y_train, x_test, y_test, max_iter=1000, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("linear_svc", LinearSVC(max_iter=max_iter))
    ])

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if verbose:
        common.print_confusion_matrix(y_test, prediction)
        common.print_classification_rprt(y_test, prediction)

    # Return the best model
    acc_score, train_score, test_score = common.print_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy_score": acc_score, "train_score": train_score, "test_score": test_score}


def svc_model(preprocessor, x_train, y_train, x_test, y_test, max_iter=1000, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svc", SVC(max_iter=max_iter))
    ])

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if verbose:
        common.print_confusion_matrix(y_test, prediction)
        common.print_classification_rprt(y_test, prediction)

    # Return the best model
    acc_score, train_score, test_score = common.print_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy_score": acc_score, "train_score": train_score, "test_score": test_score}
