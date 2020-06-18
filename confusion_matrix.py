from __future__ import division
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("Could Not import matplotlib")

def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title="Confusion Matrix",
                        cmap=None,
                        plot = True):
    '''
    plots the given confusion matrix
    :param cm: confusion matrix
    :param classes: list of unique classes
    :param normalize: True if elements in cm are of float type
    :param title: Title of the plot
    :param cmap:
    :param plot: True if a plot is to be generated
    :return:
    '''

    try:
        if cmap is None:
            cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='Predicted label',
               xlabel='True label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if plot:
            plt.show()

    except Exception as e:
        print("Could not generate graphical plot, continuing anyway!", e)
        return None,classes, cm

    return ax,classes, cm

def confusionMatrix(actual, predicted):
    '''
    calculates the confusion matrix from true labels and predicted labels
    :param actual: true labels
    :param predicted: true labels
    :return: confusion matrix
    '''
    actual = list(actual)
    predicted = list(predicted)
    assert len(actual)==len(predicted)
    classes = np.unique(actual).tolist()
    num_classes = len(classes)
    cm = np.zeros((num_classes,num_classes), dtype=int)
    for i in range(len(predicted)):
        # actual label along the rows and predicted label along the column
        x = classes.index(predicted[i])
        y = classes.index(actual[i])
        cm[x][y] +=1
    return cm, classes

def classAccuracy(cm, classes):
    '''
    calculates accuracy and error for each class
    :return: a numpy array of class accuracies
    '''

    row, col = cm.shape
    num_examples = np.sum(cm, axis=0)
    num_correct_predictions = []

    for i in range(row):
        for j in range(col):
            if i==j:
                num_correct_predictions.append(cm[i][j])

    class_acc = num_correct_predictions / num_examples
    return class_acc, classes

def overallAccuracy(true_labels, predictions):
    '''
    calculates accuracy
    :param true_labels:
    :param predictions:
    :return: returns accuracy
    '''
    true_labels = list(true_labels)
    result = list(map(lambda x,y: (1 if str(x)==str(y) else 0), predictions, true_labels))
    accuracy = sum(result)/(len(result))
    return accuracy

if __name__ == '__main__':
    # P = ['a', 'b', 'c', 'a', 'b']
    # A = ['a', 'b', 'b', 'c', 'a']
    # example found on the web.
    A = ['Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
    P = ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa']

    cm, classes = confusionMatrix(A, P)
    accuracy, _ = classAccuracy(cm, classes)
    print(classes)
    print(accuracy)
    plotConfusionMatrix(cm.astype(float), classes, normalize=True)