import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, camera2
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_1_essential_matrix import essentialMatrix
from q3_2_triangulate import triangulate, findM2
from q4_2_visualize import compute3D_pts
import scipy

# Insert your package here

# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    print('function run')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''

def dist_to_epipolarline(pts1_homo_t, pts2_homo_t, F):
    epipolar_lines = (pts2_homo_t @ F)
    # Calculate the distance between pts1 and epipolar line of pt2 on im1
    dist = np.abs(np.sum(epipolar_lines * pts1_homo_t, axis=1)) / (epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2)**0.5
    
    return np.abs(dist)

def ransacF(pts1, pts2, M, nIters=1000, tol=2.0):
    # Replace pass by your implementation
    
    N = pts1.shape[0]
    pts1_homo_t = np.concatenate((pts1, np.ones((N, 1))), axis=1)
    pts2_homo_t = np.concatenate((pts2, np.ones((N, 1))), axis=1)

    max_inlier_num = 0
    best_inlier_idx = None
    for i in range(nIters):
        idx_selected = np.random.choice(N, 7, replace=False)
        pts1_selected = pts1[idx_selected, :]
        pts2_selected = pts2[idx_selected, :]
        
        #Try sevenpoint algorithm
        Farray = sevenpoint(pts1_selected, pts2_selected, M)
        for F in Farray:
            dist_epipolarline2_pts1 = dist_to_epipolarline(pts1_homo_t, pts2_homo_t, F)
            inlier_idx = np.where(np.abs(dist_epipolarline2_pts1) < tol)[0]
            if inlier_idx.shape[0] > max_inlier_num:
                max_inlier_num = inlier_idx.shape[0]
                best_inlier_idx = inlier_idx
                print('Inlier number: {}'.format(max_inlier_num))
        
        if i == 0:
            dist_epipolarline2_pts1 = dist_to_epipolarline(pts1_homo_t, pts2_homo_t, F)
            print("Distances before ransac: {0}".format(np.abs(dist_epipolarline2_pts1)))
        
    #Run sevenpoint algorithm with the best inliers
    pts1_best, pts2_best = pts1[best_inlier_idx, :], pts2[best_inlier_idx, :]
    Farray = sevenpoint(pts1_best, pts2_best, M)
    
    #Find best F maximizing inliers within Farray
    F_best = None
    max_inlier_num = 0
    for F in Farray:
        epipolar_lines = (pts2_homo_t @ F)
        dist_epipolarline2_pts1 = dist_to_epipolarline(pts1_homo_t, pts2_homo_t, F)
        inlier_idx = np.where(np.abs(dist_epipolarline2_pts1) < 2.0)[0]
        if inlier_idx.shape[0] > max_inlier_num:
            max_inlier_num = inlier_idx.shape[0]
            F_best = F
    
    dist_epipolarline2_pts1 = dist_to_epipolarline(pts1_homo_t, pts2_homo_t, F)
    print("Distances after ransac: {0}".format(np.abs(dist_epipolarline2_pts1)))
    
    return F_best, best_inlier_idx

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    eps = 0.001
    theta = np.sum(r**2)**0.5
    if np.abs(theta) < eps:
        return np.eye(3, dtype=np.float32)
    else:
        u = r / theta
        u1, u2, u3 = u[0, 0], u[1, 0], u[2, 0]
        # u1, u2, u3 = u[0], u[1], u[2]
        u_cross = np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]], dtype=np.float32)
        
        R = np.eye(3, dtype=np.float32) * np.cos(theta) \
            + (1 - np.cos(theta)) * (u @ u.transpose()) \
            + u_cross * np.sin(theta)
            
        return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''

## additionally defined functions
def eq(a, b):
    eps = 0.001
    return np.abs(a - b) < eps

def gt(a, b):
    eps = 0.001
    return a - b > eps

def S_half(r):
    length = np.sum(r**2)**0.5
    r1, r2, r3 = r[0, 0], r[1, 0], r[2, 0]
    if (eq(length, np.pi) and eq(r1, r2) and eq(r1, 0) and gt(0, r3)) \
        or (eq(r1, 0) and gt(0, r2)) \
        or (gt(0, r1)):
        return -r
    else:
        return r

def arctan2(y, x):
    if gt(x, 0):
        return np.arctan(y / x)
    elif gt(0, x):
        return np.pi + np.arctan(y / x)
    elif eq(x, 0) and gt(y, 0):
        return np.pi*0.5
    elif eq(x, 0) and gt(0, y):
        return -np.pi*0.5

def invRodrigues(R):
    # Replace pass by your implementation
    eps = 0.001
    A = (R - R.transpose())*0.5
    a32, a13, a21 = A[2, 1], A[0, 2], A[1, 0]
    rho = np.array([[a32], [a13], [a21]], dtype=np.float32)
    s = np.sum(rho**2)**0.5
    c = (R[0, 0]+R[1, 1]+R[2, 2] - 1) / 2.0
    if eq(s, 0) and eq(c, 1):
        return np.zeros((3, 1), dtype=np.float32)
    elif eq(s, 0) and eq(c, -1):
        V = R+np.eye(3, dtype=np.float32)
        # find a nonzero column of V
        mark = np.where(np.sum(V**2, axis=0) > eps)[0]
        v = V[:, mark[0]]
        u = v / (np.sum(v**2)**0.5)
        r = S_half(u*np.pi)
        return r
    elif not eq(s, 0):
        u = rho / s
        theta = arctan2(s, c)
        return u*theta

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
## additionally defined functions
def flatten(P, r2, t2):
    # P: (N, 3)
    # r2: (3, 1)
    # t2: (3, 1)
    # (3+3+N*3,)
    return np.concatenate((r2.reshape(-1), t2.reshape(-1), P.reshape(-1)), axis=0)

def inflate(x):
    r2 = x[0:3].reshape(-1, 1)
    t2 = x[3:6].reshape(-1, 1)
    P  = x[6:].reshape(-1, 3)
    return P, r2, t2

#Addi functions

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    P, r2, t2 = inflate(x)
    R2 = rodrigues(r2)
    M2 = np.concatenate((R2, t2), axis=1)
    points_word_homo = np.concatenate( ( P, np.ones( (P.shape[0], 1) ) ), axis=1 ).transpose()
    points_img1_rep_homo = K1 @ M1 @ points_word_homo
    points_img1_rep = points_img1_rep_homo[0:2, :] / points_img1_rep_homo[2, :]
    points_img2_rep_homo = K2 @ M2 @ points_word_homo
    points_img2_rep = points_img2_rep_homo[0:2, :] / points_img2_rep_homo[2, :]

    error_img1_rep = (p1 - points_img1_rep).reshape(-1)
    error_img2_rep = (p2 - points_img2_rep).reshape(-1)

    residuals = np.concatenate((error_img1_rep, error_img2_rep), axis=0)
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    p1 = p1.transpose()
    p2 = p2.transpose()
    
    residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)    
    R2_init = M2_init[:, 0:3]
    t2_init = M2_init[:, 3]
    r2_init = invRodrigues(R2_init)
    x_init = flatten(P_init, r2_init, t2_init)
    x_optim, _ = scipy.optimize.leastsq(residual, x_init)
    
    # print('Reprojection error after BA: %f' % np.sum(residual(x_optim)**2))

    P2, r2, t2 = inflate(x_optim)
    R2 = rodrigues(r2)
    M2 = np.concatenate((R2, t2), axis=1)
    
    obj_start = rodriguesResidual(K1, M1, p1, K2, p2, x_init)
    obj_end = rodriguesResidual(K1, M1, p1, K2, p2, x_optim)

    # print('Object start: {}'.format(obj_start))
    # print('Object end: {}'.format(obj_end))

    return M2, P2, obj_start, obj_end

if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    
    noisy_pts1, noisy_pts2 = noisy_pts1[inliers, :], noisy_pts2[inliers, :]

    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)

    # M1, M2, camera extrinsics
    # let camera 1 be the center of the world
    M1 = np.zeros((3, 4), dtype=np.float32)
    M1[[0, 1, 2], [0, 1, 2]] = 1
    # obtain four possible M2s from E
    M2s = camera2(E)
    print(M2s.shape)

    C1 = K1 @ M1

    # recovered point clouds, (N, 3)
    Ps = list()
    C2s = list()
    chose_M2_idx = 2
    # get best M2
    for i in range(M2s.shape[2]):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        C2s.append(C2)
        P, cur_err = triangulate(C1, noisy_pts1, C2, noisy_pts2)
        if np.min(P[:, 2]) > 0:
            chose_M2_idx = i
        Ps.append(P)
        print('Reprojection error of M2_%d: %f' % (i, cur_err))

    # choose a best M2
    # chose_M2_idx = 2
    print('chosen M2 idx: %d' % chose_M2_idx)
    M2 = M2s[:, :, chose_M2_idx]
    P_before = Ps[chose_M2_idx]
    C2 = C2s[chose_M2_idx]

    _, cur_err = triangulate(C1, noisy_pts1, C2, noisy_pts2)
    print('Reprojection error of M2 before BA: %f' % (cur_err))

    # find M2 ============
    # bundle adjustment

    M2_BA, P_BA, _, _ = bundleAdjustment(K1, M1, noisy_pts1, K2, M2, noisy_pts2, P_before)
    # YOUR CODE HERE
    
    C2_BA = K2 @ M2_BA
    _, cur_err = triangulate(C1, noisy_pts1, C2_BA, noisy_pts2)
    print('Reprojection error of M2_BA: %f' % (cur_err))

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    F /= F[2, 2]
    
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    
    # YOUR CODE HERE

    # Simple Tests to verify your implementation:
    # from scipy.spatial.transform import Rotation as sRot
    # rotVec = sRot.random()
    # mat = rodrigues(rotVec.as_rotvec())

    # assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    # assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    plot_3D_dual(P_before, P_BA)
    # YOUR CODE HERE