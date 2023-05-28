from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def LDA(X, y, n):
    # X = data
    # y = labels
    # n = number of components
    col = X.shape[1]
    classes = np.unique(y)

    # print(classes)
    # Within class scatter matrix:
    # SW = sum((x - mean_x)^2 )

    # Between class scatter:
    # SB = sum( n_c * (mean_x - meanArr)^2 )

    meanArr = np.mean(X, axis=0)
    SW = np.zeros((col, col))
    SB = np.zeros((col, col))
    for c in classes:
        x = X[y == c]
        mean = np.mean(x, axis=0)
        # (4, n_c) * (n_c, 4) = (4,4) -> transpose
        SW += (x - mean).T.dot((x - mean))

        # (4, 1) * (1, 4) = (4,4) -> reshape
        n_c = x.shape[0]
        mean_diff = (mean - meanArr).reshape(col, 1)
        SB += n_c * (mean_diff).dot(mean_diff.T)

    # Determine SW^-1 * SB
    # print(SW)
    # print(SB)
    A = np.linalg.inv(SW).dot(SB)
    # Get eigenvalues and eigenvectors of SW^-1 * SB
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # -> eigenvector v = [:,i] column vector, transpose for easier calculations
    # sort eigenvalues high to low
    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    # print(eigenvalues)
    eigenvectors = eigenvectors[idxs]
    # print(eigenvectors)
    # store first n eigenvectors
    linear_discriminants = eigenvectors[0:n]
    # print(self.linear_discriminants)
    return np.dot(X, linear_discriminants.T)


iris = datasets.load_iris()

# print(dataset['data'])
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# print(iris_df.describe())
x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

# print(x.head())
# scaler = StandardScaler()
# print(x.head())
# print(y.head())

x = np.asarray(x)
y = np.asarray(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# print(x_train)
# print(y_train)


print(
    f"training set size: {x_train.shape[0]} samples, Test set size: {x_test.shape[0]} samples"
)

knn = KNeighborsClassifier(5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(f"Accuracy without LDA: {accuracy_score(y_test, y_pred)}")


x = LDA(x, y, 2)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(5)
knn.fit(x_train, y_train)
y_predlda = knn.predict(x_test)
print(f"Accuracy with LDA: {accuracy_score(y_test, y_predlda)}")


# YES, Using LDA as the preprocessing step increases accuracy of KNN. However, other factors such as the number of neighbors, the number of components, and the test size can also affect the accuracy of KNN.
