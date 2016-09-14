from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    return (X_train, labels_train), (X_test, labels_test)

def train(X_train, y_train, reg=1):
    ''' Build a model from X_train -> y_train '''
    # y_train is 60000 x 10
    # x_train is 60000 x 786
    # left is 10 x 786
    # right is 786 x 786
    y_train = one_hot(y_train)
    right = np.zeros((X_train.shape[1], y_train.shape[1]))
    left = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(X_train.shape[0]):
        if i % 1000 == 0:
            print(i)
        right += np.outer(X_train[i], np.transpose(y_train[i]))
        left += np.outer(X_train[i], np.transpose(X_train[i]))
    left = left + reg*np.identity(X_train.shape[1])
    left = np.linalg.inv(left)
    return np.dot(left, right)

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    result = np.zeros((labels_train.shape[0], NUM_CLASSES))
    for i in range(labels_train.shape[0]):
        result[i][labels_train[i]] = 1;
    return result

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    results = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        results[i] = np.argmax(np.dot(np.transpose(model), X[i]))
    
    return results

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))