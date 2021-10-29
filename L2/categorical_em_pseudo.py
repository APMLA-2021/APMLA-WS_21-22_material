import numpy as np


class CategoricalEM:
    def __init__(self, K, I, N, delta, epochs, init_params):
        self.K = K  # Number of mixture components
        self.I = I
        self.N = N
        self.delta = delta  # Mininum increment in the log-likelihood
        self.epochs = epochs  # Maximum number of iterations
        self.init_params = init_params

        self.r_matrix = np.zeros([N, K])  # [N, K]

        self.theta_matrix = np.random.dirichlet(init_params['theta'] * np.ones(I), size=K)  # [K, I]
        self.Q_list = list()

        if init_params['pi'] > 0:
            self.pi_vector = np.random.dirichlet(init_params['pi'] * np.ones(K), size=1)  # [1, K]
        else:
            self.pi_vector = 1 / K * np.ones([1, K])

    def fit(self, X):
        i = 0
        prev_Q = 0
        self.Q_list = list()
        while (i + 1 <= self.epochs):
            Q, self.r_matrix = self.E_step(X, self.pi_vector, self.theta_matrix)
            self.pi_vector, self.theta_matrix = self.M_step(self.r_matrix, X)
            diff = Q - prev_Q if i > 0 else 200
            self.Q_list.append(Q)
            prev_Q = Q
            if (diff < self.delta):
                str_tmp = 'ITER: ' + str(i) + ' Q= ' + str(np.around(Q, 4)) + ' diff= ' + str(np.around(diff, 4))
                print(str_tmp)
                break
            if (i % 5 == 0):
                str_tmp = 'ITER: ' + str(i) + ' Q= ' + str(np.around(Q, 4)) + ' diff= ' + str(np.around(diff, 4))
                print(str_tmp)
            i = i + 1

    def E_step(self, X, pi_vector, theta_matrix):
        # FILL HERE
        return Q, r_matrix

    def rik_matrix(self, X, pi_vector, theta_matrix):
        # FILL HERE
        return np.clip(r_matrix, 1e-250, None), last_term2

    # Log sum exponential trick
    def compute_lset(self, ns):
        max_ = np.max(ns)
        ds = ns - max_
        sumOfExp = np.exp(ds).sum()
        return max_ + np.log(sumOfExp)

    def M_step(self, r, X):
        # FILL HERE
        return pi_vector, theta_matrix
