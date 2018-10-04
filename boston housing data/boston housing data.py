from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    print(boston['DESCR'])
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.xlabel(features[i])
        plt.ylabel('Med House Price')
        plt.scatter(X[:, i], y)
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    a = np.dot(np.transpose(X), X)
    b = np.dot(np.transpose(X), Y)
    w = np.linalg.solve(a, b)
    return w

def MSE(w,X,Y):
    n = X.shape[0]
    d = X.shape[1]
    Y_hat = np.dot(X, w)
    e = Y - Y_hat
    SSE = np.dot(e.T, e)
    return SSE/(n-d)
    
def MAE(w,X,Y):
    n = X.shape[0]
    Y_hat = np.dot(X, w)
    e = np.abs(Y - Y_hat)
    return np.sum(e)/n
    
def R2(w,X,Y):
    Y_hat = np.dot(X, w)
    e = Y - Y_hat
    SSE = np.dot(e.T, e)
    SST = np.sum((Y-np.mean(Y))**2)
    R2 = 1 - SSE/SST
    return R2

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    print('number of data points: {}'.format(X.shape[0]))
    print('number of features: {}'.format(X.shape[1]))
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    X = np.c_[np.ones(506),X ] #add column 1's as last column
    training_index = np.random.choice(506, 405, replace=False)
    test_index  = np.setdiff1d(np.arange(0, X.shape[0]), training_index)
    traing_X = X[training_index, :]
    traing_y = y[training_index]
    test_X = X[test_index, :]
    test_y = y[test_index]
    # Fit regression model
    w = fit_regression(traing_X, traing_y)
    
    for i in range(features.shape[0]):
        print(features[i] + ': {}'.format(w[i+1]))
    print('')
    
    # Compute fitted values, MSE, etc.
    mse = MSE(w, test_X, test_y)
    print("MSE for test set: {}".format(mse))
    mae = MAE(w, test_X, test_y)
    print("MAE for test set: {}".format(mae))
    r2 = R2(w, test_X, test_y)
    print("R2 for test set: {}".format(r2))
    print('')
    
    #feature selection
    for j in range(features.shape[0]):
        w = fit_regression(traing_X[:, [0,j+1]], traing_y)
        mse = MSE(w, test_X[:,[0,j+1]], test_y)
        mae = MAE(w, test_X[:,[0,j+1]], test_y)
        r2 = R2(w, test_X[:,[0,j+1]], test_y)
        print(features[j] + ': w={}, MSE={}, MAE={} R2={}'.format(w[1], mse, mae, r2))
    
if __name__ == "__main__":
    main()

