# #################################################################################################################### #
#       da.py                                                                                                          #
#           Models based on discriminant analysis.                                                                     #
# #################################################################################################################### #

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from . import common


def linear_discriminant_model(preprocessor, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("linear_discriminant", LinearDiscriminantAnalysis())
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
