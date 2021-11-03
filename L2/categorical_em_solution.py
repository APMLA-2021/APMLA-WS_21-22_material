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
        log_pi_vector = np.log(pi_vector)

        r_matrix, last_term = self.rik_matrix(X, pi_vector, theta_matrix)  # [N, K]

        r_matrix = r_matrix / np.tile(np.sum(r_matrix, 1, keepdims=True), [1, r_matrix.shape[1]])
        log_pi_matrix = np.tile(log_pi_vector, [self.N, 1])
        term1 = np.multiply(r_matrix, log_pi_matrix)
        term1 = np.sum(term1)

        Q = term1 + np.sum(np.multiply(r_matrix, last_term))
        return Q, r_matrix

    def rik_matrix(self, X, pi_vector, theta_matrix):

        log_theta_matrix = np.log(theta_matrix)
        log_pi_vector = np.log(pi_vector)

        last_term = np.zeros([self.N, 1])
        last_term2 = X @ log_theta_matrix.T
        ns = last_term2 + np.tile(log_pi_vector, [self.N, 1])  # [N, K]
        for i in range(self.N):
            last_term[i] = self.compute_lset(ns[i, :])

        last_term = np.tile(last_term, [1, self.K])
        log_r_matrix = np.tile(log_pi_vector, [self.N, 1]) + last_term2 - last_term
        r_matrix = np.exp(log_r_matrix)

        return np.clip(r_matrix, 1e-250, None), last_term2

    # Log sum exponential trick
    def compute_lset(self, ns):
        max_ = np.max(ns)
        ds = ns - max_
        sumOfExp = np.exp(ds).sum()
        return max_ + np.log(sumOfExp)

    def M_step(self, r, X):

        pi_vector = np.mean(r, 0)

        # Update of theta_matrix
        num = r.T @ X  # [K, I]

        den = np.tile(np.sum(num, 1, keepdims=True), [1, self.I])

        theta_matrix = np.divide(num, den)
        return pi_vector, theta_matrix
