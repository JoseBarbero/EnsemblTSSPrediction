from sklearn import svm

# https://github.com/scikit-learn/scikit-learn/commit/4aea36498205efac1991056bb2beef66beab57d7
def svm_string_kernel():
    clf = svm.SVC()


if __name__ == "__main__":
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    
    clf.fit(X, y)
    SVC()