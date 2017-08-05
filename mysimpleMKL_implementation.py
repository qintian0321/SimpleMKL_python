#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      chris
#
# Created:     10/07/2017
# Copyright:   (c) chris 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

"""
kernel_matrices: Stores all the individual kernel matrices computed from input x matrix (row as sample, column as feature value)
each kernel matrix row and column combination is ordered in same order of points as y, e.g. Kij, i th sample and jth sample kernel value
returns weights and combined weighted kernel estimated from training data

"""

import numpy as np
import working_helpers_al as helpers
import working_kernel_helpers as k_helpers

"""
d_init: numpy array of size len(kernel_matrices) summing up to 1
kernel_matrices: list of kernel matrices
C: regularization parameter used in SVM
y: label value (1 or -1) for each training point

stopping criterion: duality gap
"""
#main simpleMKL SVM fitting algorithm defined here
#Obj function:Min 0.5*(sum(1/d_k*t(Vk)*Vk))+C*sum(e_i),subject to SVM constraints and sum(dk)=1, dk>=0
#Geometric interpretation: maximize the margin between positive and negative samples with specific linear combination of kernels

#Input parameter:
#d_init:initial kernel weight value vector d={d1,d2...dk} as a starting point for search.
#kernel_matrices: all pre-computed kernel matrices based on input sample matrix X which has been summarized by kernel function
#C:penalty value => 0<=alpha_i<=C
#y:label of each sample, 1/-1
#verbose: should we output the d vector update in each iteration?
def find_kernel_weights(d_init,kernel_matrices,C,y,verbose):
    ##########################################Initialization, starting from a point d
    weight_precision=1e-08 #weights below this value are set to 0
    goldensearch_precision=1e-01
    goldensearch_precision_init=1e-01
    max_goldensearch_precision=1e-08
    duality_gap_threshold=0.01#search stopping criteria defined in the paper
    for m in kernel_matrices:
        assert m.shape == (y.shape[0],y.shape[0])
    M = len(kernel_matrices)#how many kernels we have
    d = d_init#initial guessed weights of each kernel, d_m=1/M, where M is the number of kernels
    y_mat = np.outer(y, y)#Creates y matrix for use in SVM later
    iteration = 0
    stop_state=False #loop parameter
    ##########################################ALgorithm 1 pseudocode defined in the simpleMKL paper:
    #stop_state: check the dual gap between the primal MKL and dual MKL in each loop
    #d: weighted vector d={d1, d2 ... dm}, m=1,2...M kernels
    #dJ: gradient vector computed on current d vector
    #D: reduced gradient descent direction vector computed based on dJ and equality constraint
    while(not stop_state):#while loop until minimizers of d and corresponding alphas are found
        if verbose==1:
            print "iteration:",iteration
            print "d:",d
        old_d=d.copy()
        #########################################SVM computation to get current d value and J(d) value
        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices,d)#given current d vector value, compute the combined kernel matrices value
        alpha,J=helpers.compute_J_SVM(combined_kernel_matrix, y_mat,C)#SVM wrapper to solve alphas given current d and J(d) value at this point
        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)#compute current GRADIENT of J(d),m-dimension vector given alpha values
        mu = np.argmax(d)#mu is the index of the largest component in d vector
        D = helpers.compute_reduced_descent_direction(d,dJ,mu)#compute the REDUCED gradient direction based on equality constraint on current d vector
        if verbose==1:
            print 'current gradient:'
            print dJ
            print 'current alpha, J: ',alpha, J
            print 'current reduced descent: ',D
        J_cross=0
        d_cross=d
        D_cross=D
        counter=1
        J_prev=J
        while (J_cross < J):#an efficient update of d vector without the need to recompute the gradient at each new d vector
            #update d vector, D vector to d_cross and D_cross and corresponding J_cross value
            d=d_cross#update d
            D=D_cross#update reduced gradient
            if counter>1:#in the start of the while loop, J_cross = 0 and will lead to a bug
                J=J_cross#update function value
            #compute the maximum step size based on current d and D value
            gamma_max=helpers.compute_max_admissible_gamma(d,D)#compute the largest step size based on current d and D value
            delta_max=gamma_max
            #update d_cross based on gamma_max
            d_cross = d + gamma_max*D #d_cross is a new point along the direction of reduced gradient D with step size gamma_max, so one component of d_cross will reach zero
            #update J_cross based on d_cross
            combined_kernel_matrix_cross = k_helpers.get_combined_kernel(kernel_matrices,d_cross)#combined kernel given the new d_cross vector
            alpha_cross,J_cross= helpers.compute_J_SVM(combined_kernel_matrix_cross,y_mat,C)#compute the SVM solution with the d_cross vector
            if J_cross < J:#only update D_cross when J_cross < J(d)
                D_cross=helpers.update_reduced_descent_direction(d_cross,D,mu,weight_precision)#update the reduced gradient direction when the function value keeps decreasing
                counter=counter+1
            if verbose==1:
                print "updated cost: ",J_cross
                print "d cross:"
                print d_cross
                print "counter:",counter
                print "updated D_cross:"
                print D_cross
        #Now J(d_cross) > J(d), keep in mind that d has been updated several times before.
        #Do line-search along direction of D (no further update) to obtain the point which minimizes function value J between point d and point d_cross.
        gamma, alpha, J=helpers.compute_gamma_linesearch(0,gamma_max,
                                                delta_max,J,J_cross,
                                               d,D,kernel_matrices,
                                               J_prev,y_mat,alpha,C,
                                               goldensearch_precision)
        d = d + gamma * D #update d to the new point to further decrease J value, gamma might be zero, i.e no update
        # numerical cleaning
        d = helpers.fix_weight_precision(d,weight_precision)
        # improve line search by enhancing precision
        if max(abs(d-old_d))<weight_precision and goldensearch_precision>max_goldensearch_precision:
            goldensearch_precision=goldensearch_precision/10
        dJ_curr_d = helpers.compute_dJ(kernel_matrices, y_mat, alpha)#compute the gradient value at the new point d and corresponding alpha values
        # stopping criterion: check difference between primal J(d) and dual function value
        duality_gap=(J+np.max(-dJ_curr_d) -np.sum(alpha))/J
        print 'duality gap: ',duality_gap
        if duality_gap<duality_gap_threshold:
            stop_state=True
        iteration += 1
    return (d,k_helpers.get_combined_kernel(kernel_matrices, d),J,alpha,duality_gap)