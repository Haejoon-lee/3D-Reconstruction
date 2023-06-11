import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import triangulate
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence
from helper import camera2
from q3_1_essential_matrix import essentialMatrix

# Insert your package here

'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points.
'''

def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # YOUR CODE HERE
    coords_x1 = temple_pts1['x1']
    coords_y1 = temple_pts1['y1']

    coords_x2 = coords_y2 = []
    
    # Find x2, y2
    for i in range(coords_x1.shape[0]):
        x1, y1 = coords_x1[i, 0], coords_y1[i, 0]
        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
        coords_x2.append(x2)
        coords_y2.append(y2)

    coords_x2, coords_y2 = np.array(coords_x2).reshape(-1, 1), np.array(coords_y2).reshape(-1, 1)

    pts1 = np.concatenate((coords_x1, coords_y1), axis=1)
    pts2 = np.concatenate((coords_x2, coords_y2), axis=1)

    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)

    # Assume camera coordinates = word coordinates
    M1 = np.zeros((3, 4), dtype=np.float32)
    M1[0,0] = M1[1,1] = M1[2,2] = 1   
    
    C1 = K1 @ M1

    # Find C2, P for each M2
    P_list = []
    C2_list = []
    err_list = []
    for i in range(M2s.shape[2]):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2
        C2_list.append(C2)
        
        P, err_rep = triangulate(C1, pts1, C2, pts2)
        P_list.append(P)
        
        print('Reprojection error of M2_%d: %f' % (i, err_rep))
        err_list.append(err_rep)

    #Remain best M2
    min_idx = np.argmin(np.abs(np.array(err_list)))
    # min_idx = 2

    print(min_idx)

    M2 = M2s[:, :, min_idx]
    P = P_list[min_idx]
    C2 = C2_list[min_idx]

    np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

    return P

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    P = compute3D_pts(temple_coords_path, intrinsics, F, im1, im2)

    #Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
    ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
    zmin, zmax = np.min(P[:, 2]), np.max(P[:, 2])

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o')

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(im1)
    
    # plt.subplot(122)
    # plt.imshow(im2)
    
    plt.show()
