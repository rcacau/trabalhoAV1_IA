import numpy as np



class PolynomialRegression:
    def __init__(self, X_train, Y_train, q=1):
        self.X = np.copy(X_train)
        self.N, self.p = X_train.shape
        self.q = q
        for i in range(q):
            if i == 0:
                x = np.ones((self.N,1))
            else:
                x = X_train**(i+1)
            self.X = np.hstack((
                self.X, x
            ))
        self.Y = Y_train

    def fit(self):
        # beta1 = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        self.beta = np.linalg.pinv(self.X)@self.Y
        # beta3 = np.linalg.lstsq(self.X, self.Y)[0]
    def predict(self, X):
        if len(X.shape) < 3:
            N, p = X.shape
            Xt = np.copy(X)
            for i in range(self.q):
                if i == 0:
                    x = np.ones((N,1))
                else:
                    x = X**(i+1)
                Xt = np.hstack((
                    Xt, x
                ))
        else:
            z,k,p = X.shape
            Xt = np.copy(X)
            for i in range(self.q):
                if i == 0:
                    x = np.ones((z,k,1))
                else:
                    x = X**(i+1)
                Xt = np.concatenate((
                    Xt, x
                ),axis=2)

        return Xt @ self.beta
        



        

class LinearRegression:
    def __init__(self, X_train, Y_train, solver = 'OLS', fit_intercept = True) -> None:

        self.X = X_train
        self.N, self.p = X_train.shape
        self.fit_intercept = fit_intercept
        if fit_intercept:
            self.X = np.hstack((
                np.ones((self.N,1)), X_train
            ))
        self.Y = Y_train
    def fit(self):
        self.beta = np.linalg.inv(self.X.T @ self.X)@self.X.T@self.Y
    

    def predict(self, X):
        N, p = X.shape
        Xt = np.copy(X)
        if self.fit_intercept:
            Xt = np.hstack((
                np.ones((N,1)), Xt
            ))
        
        return Xt @ self.beta