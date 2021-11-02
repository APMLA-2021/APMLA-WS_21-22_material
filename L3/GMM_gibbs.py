#!/usr/bin/env python
# Created by Joe Ellis 
# Columbia University DVMM Lab 

# STD Libraries
import os

# Numerical Computing Libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class MixGaussGibbsSampler():
    def __init__(self, X, Y, sigma=1, lam=1, burn_in=50, lag=10):
        """ This function initializes the Gibbs sampler with some standard parameters, 
        for how the Gibbs Sampler works
        burn_in = The number of iterations that we reach before assumed convergence 
                  to the stationary distribution.
        lag = The number of iterations between samples once we have reached the stationary distribution
        X = The random variables.
        Y = The randomly generated labels for the variables
        sigma = model parameter
        lam = model parameter
        """

        # Add the variables to the class
        self.X = X # The data points
        self.Y = Y # The cluster assignments, this should be generated randomly
        self.burn_in = burn_in
        self.lag = lag
        self.sigma = 1.0
        self.lam = 1.0
        self.num_mixtures = self.Y.max() + 1
    
        self.u_locations = np.zeros((self.num_mixtures, X.shape[1]))
        self.pi_ks = np.ones(self.num_mixtures)/ float(self.num_mixtures)
        #self.s_vec = np.zeros(3)
        #self.zeta = np.zeros(3)

        # Constant variable
        self._normalizer = (1 / ((self.sigma ** 2) * np.sqrt(2 * np.pi)))

        self.iter_prob = [] # A list variable holding the total probability
                            # at each stage of the iteration
        pass

    def perform_gibbs_sampling(self, iterations=False):
        """ This function controls the overall function of the gibbs sampler, and runs the gibbs
        sampling routine.
        iterations = The number of iterations to run, if not given will run the amount of time 
                     specified in burn_in parameter
        """
        if not iterations:
            num_iters = self.burn_in
        else:
            num_iters = iterations

        # Plot the initial set up
        self.plot_points("Initial Random Assignments")

        # Run for the given number of iterations
        for i in range(num_iters):
            self.calc_dist_log_prob()
            self.sample_mixture_locations()
            self.sample_mixture_assignments()

        # Plot the final mixture assignments
        self.plot_points("Final Mixture Assignments Algorithm 2")

        # Final Plot of the log probability as a function of iteration
        self.plot_prob()

        return self.u_locations, self.Y

    def sample_mixture_locations(self):
        """ This function samples from the mixture locations and updates them"""
        for i in range(0,self.num_mixtures):
            
            # Assigned indices
            assigned_indices = (self.Y == i)
            zeta = assigned_indices.sum()

            # used x and create the xbar
            class_x = self.X[assigned_indices, :]
            x_bar = np.sum(class_x, axis=0)
            u_bar = x_bar / float(zeta)

            # Now let's create the hat distributions
            lambda_hat = ((zeta/(self.sigma ** 2)) + (1/self.lam ** 2)) ** -1
            #print lambda_hat
            #print u_bar
            u_hat = ((zeta / (self.sigma ** 2)) * lambda_hat) * u_bar

            # Now sample this mixture location this assumes non covariance
            # just a simple standard distibution
            for n in range(0, self.X.shape[1]):
                self.u_locations[i,n] = np.random.normal(u_hat[n], lambda_hat)

        pass

    def sample_mixture_assignments(self):
        """ Now we will sample the cluster assignments given the given mixture locations"""

        for i in range(0,self.X.shape[0]):
            # Sample the cluster assignment
            self.sample_mixture_assignment(i)

        pass

    def sample_mixture_assignment(self, i):
        """ This function performs one sampling assignment
        i = The assignment index to be samples"""
        probs = np.zeros(self.num_mixtures)
        for k in range(0, self.num_mixtures):
            #log_prob[k] = np.log(norm.pdf(self.X[i, :], self.u_locations, self.sigma ** 2))
            d = np.linalg.norm(self.X[i, :] - self.u_locations[k, :])
            #print d
            #print np.exp(-(d ** 2) / (2 * (self.sigma ** 2)))
            probs[k] = np.exp(-(d ** 2) / (2 * (self.sigma ** 2))) * self._normalizer
            #print probs[k]
        # Now let's look at the normed probabilities
        norm_probs = probs / float(sum(probs))

        # Now let's create a list variable that holds each value up to 1 so we can use a 
        # uniform random number generator to sample from the cluster assignments
        samp_divs = [norm_probs[0], sum(norm_probs[0:2])]
        rand_num = np.random.rand(1)

        # Now get the cluster assignment
        cluster_assign = False
        for k in range(0,self.num_mixtures - 1):
            if rand_num < samp_divs[k]:
                cluster_assign = k
                break

        # If it's larger than all of our sample divs then assign last mixture to our variable
        if cluster_assign is False:
            cluster_assign = self.num_mixtures - 1

        self.Y[i] = cluster_assign
        pass

    def calc_dist_log_prob(self):
        """ This function calcuates the total log probability of the given 
        function value, with the current categorical assignments over the data
        points and the gaussian means"""
        log_probs = np.zeros(self.X.shape[0])
        for i in range(0, self.X.shape[0]):
            # Calculate the sum of the lob probabilities
            clust = self.Y[i]
            d = np.linalg.norm(self.X[i, :] - self.u_locations[clust, :])
            #print d
            #print np.exp(-(d ** 2) / (2 * (self.sigma ** 2)))
            log_probs[i] =  np.log(np.exp(-(d ** 2) / (2 * (self.sigma ** 2))) * self._normalizer)

        self.iter_prob.append(log_probs.sum())
        pass

    ### Output Functions
    def plot_prob(self):
        """ This function plots the probability as a function of the 
        iterations that are completed"""

        fig, ax = plt.subplots()
        iteration_vec = range(0, len(self.iter_prob))
        ax.plot(iteration_vec, self.iter_prob)
        ax.set_title('Log Probability vs. Iterations')

        plt.show()
        pass

    def plot_points(self, title):
        """ This plots the points and the u_locations in a scatter plot"""
        fig, ax = plt.subplots()
        datasets = []
        for i in range(self.num_mixtures):
            # Assigned indices
            assigned_indices = (self.Y == i)
            datasets.append(self.X[assigned_indices, :])

        # Now let's put the scatter plots onto the scene.
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for j,data in enumerate(datasets):
            ax.scatter(data[:, 0], data[:, 1], color=colors[j])

        ax.set_title(title)

        plt.show()
        pass


#ENDCLASS
def generate_test_data():
    """ This is the a test script used for me to generate test data"""
    # All of these will be assumed 2-D data
    # Let's generate a mixture of three gaussians
    u_0 = np.array([-2.0,-2.0])
    u_1 = np.array([0.0,2.0])
    u_2 = np.array([2.0,-2.0])

    # sigmas for each of the data
    sigmas = [1, 1, 1]
    points_per_cluster = 30

    # Now initialize the data variables
    data0 = np.random.randn(points_per_cluster,2)*sigmas[0] + u_0
    data1 = np.random.randn(points_per_cluster,2)*sigmas[1] + u_1
    data2 = np.random.randn(points_per_cluster,2)*sigmas[2] + u_2
    X = np.vstack((data0, data1, data2))

    # Now random init the cluster assignments
    rand_Y = np.random.randint(0, 3, points_per_cluster*3)
    return X, rand_Y


if __name__ == "__main__":
    # Test script for checking my module
    X, Y = generate_test_data()
    # Now initalize the gibbs sampler
    gs = MixGaussGibbsSampler(X, Y)
    gs.perform_gibbs_sampling()
    # print(gs.Y)
    # print(gs.u_locations)



