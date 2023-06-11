import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:  
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''

def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    N = pts1.shape[0]

    # Normalization
    pts1, pts2 = pts1/float(M), pts2/float(M)
    
    xcoords1, ycoords1 = pts1[:, 0], pts1[:, 1]
    xcoords2, ycoords2 = pts2[:, 0], pts2[:, 1]

    # A Matix
    cul0 = xcoords2 * xcoords1
    cul1 = xcoords2 * ycoords1
    cul2 = xcoords2
    cul3 = ycoords2 * xcoords1
    cul4 = ycoords2 * ycoords1
    cul5 = ycoords2
    cul6 = xcoords1
    cul7 = ycoords1
    cul8 = np.ones((N,), dtype=np.float32)
    
    A = np.stack((cul0, cul1, cul2, cul3, cul4, cul5, cul6, cul7, cul8), axis=1)
    
    # Get F1 and F2
    _, _, Vt = np.linalg.svd(A)
    F1_vec, F2_vec = Vt[-1, :], Vt[-2, :] #(9,)
    F1, F2 = F1_vec.reshape(3, 3), F2_vec.reshape(3, 3)

    #Find the coefficients for F1 and F2 spanning the null space
    a, b = F1-F2, F2

    funct = lambda x: np.linalg.det(x*a + b)
    
    c0 = funct(0)
    c1 = (2.0/3)*(funct(1)-funct(-1)) - (1.0/12)*(funct(2)-funct(-2))
    c3 = (1.0/12)*(funct(2) - funct(-2)) - (1.0/6)*(funct(1)-funct(-1))
    c2 = funct(1) - c0 - c1 - c3
    
    #Solve the polynomial
    roots = poly.polyroots([c0, c1, c2, c3])
    
    # Unscale F
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = 1.0 / M
    T[2, 2] = 1.0

    for root in roots:
        F_norm = root*a + b
        
        F_norm = _singularize(F_norm)
        # F_norm = refineF(F_norm, pts1, pts2)
        
        F_final = T.transpose() @ F_norm @ T
        Farray.append(F_final)
    
    return Farray



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    
    indices = np.array([18, 19, 24, 54, 56, 82, 84])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    # print(Farray)

    F = Farray[2]
    F /= F[2,2]

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    F /= F[2,2]

    print("Error:", ress[min_idx])

    # print(F)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
    
    print(F)
    np.savez('q2_2.npz', F, M, pts1, pts2)
