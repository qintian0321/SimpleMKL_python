# SimpleMKL_python
Python implementation of SimpleMKL algorithm for multi-kernel SVM learning.
Adapted from the original MATLAB code: http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html.
The original paper link: https://hal.archives-ouvertes.fr/hal-00218338/fr/.
The reduced gradient steps mainly follow the pseudo-code described in Algorithm 1 in the paper, however there is a wrong update step in line 9 of the pseudo-code where the reduced gradient direction is updated when one parameter hits the boundary and their MATLAB code is correct at this step.
