import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here

'''
Q2.1: Eight Point Algorithm (Technically all points are used, not 'eight')
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''

def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
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
    
    # solve a raw f
    _, _, Vt = np.linalg.svd(A)

    F_vec = Vt[-1, :] #(9,)
    F_raw = F_vec.reshape(3, 3)

    #Refine F
    F_norm = _singularize(F_raw)
    # F_norm = refineF(F_norm, pts1, pts2) # ?
    
    # Unscale
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = 1.0 / M
    T[2, 2] = 1.0

    F_final = T.transpose() @ F_norm @ T
    
    return F_final

if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    
    indices = np.arange(pts1.shape[0])
    indices = np.random.choice(indices, 8, False)
    
    # indices = np.array([82, 19, 56, 84, 54, 24, 18])
    M=np.max([*im1.shape, *im2.shape])
    
    F = eightpoint(pts1, pts2, M)
    # F = eightpoint(pts1[indices, :], pts2[indices, :], M=np.max([*im1.shape, *im2.shape]))

    F /= F[2,2] 
    # Q2.1
    print(F)
    np.savez('q2_1.npz', F=F, M=M)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)

    displayEpipolarF(im1, im2, F)
