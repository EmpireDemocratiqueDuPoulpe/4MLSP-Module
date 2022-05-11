# #################################################################################################################### #
#       metrics.py                                                                                                     #
#           Show models metrics based on sklearn.metrics functions and matplotlib.                                     #
# #################################################################################################################### #

from matplotlib import pyplot
from sklearn import metrics


def roc_curve(model, x_test, y_test):
    prediction = model.predict(x_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    pyplot.title("Receiver Operating Characteristic (ROC)")
    pyplot.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    pyplot.legend(loc="lower right")
    pyplot.plot([0, 1], [0, 1], "r--")
    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])
    pyplot.xlabel("False Positive Rate (FPR)")
    pyplot.ylabel("True Positive Rate (TPR)")
    pyplot.show()


def precision_recall_curve(model, x_test, y_test, name):
    prd = metrics.PrecisionRecallDisplay.from_estimator(model, x_test, y_test, name=name)
    prd.ax_.set_title("2-class Precision-Recall curve")
    pyplot.show()
