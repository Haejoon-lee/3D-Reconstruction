import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''

def triangulate_3pts(C1, pts1, C2, pts2, C3, pts3):
    # Replace pass by your implementation
    coord_word_list = []
    err_reproject = 0
    
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i, :]
        x2, y2 = pts2[i, :]
        x3, y3 = pts3[i, :]
        
        # A matrix 
        A0 = C1[0, :] - x1*C1[2, :]
        A1 = C1[1, :] - y1*C1[2, :]
        A2 = C2[0, :] - x2*C2[2, :]
        A3 = C2[1, :] - y2*C2[2, :]
        A4 = C3[0, :] - x3*C3[2, :]
        A5 = C3[1, :] - y3*C3[2, :]
        A = np.stack((A0, A1, A2, A3, A4, A5), axis=0)
        
        #Get world coordinates through SVD
        U, s, Vt = np.linalg.svd(A)
        coord_word = Vt[-1, :] #(4,)
        coord_word = coord_word[0:3] / coord_word[3] #(3,)
        coord_word_list.append(coord_word)
        
        #Reprojection
        coord_word_homo = np.zeros((4, 1), dtype=np.float32)
        coord_word_homo[0:3, 0] = coord_word
        coord_word_homo[3, 0] = 1
        
        coord_cam1_rep = C1 @ coord_word_homo
        coord_cam2_rep = C2 @ coord_word_homo
        coord_cam3_rep = C3 @ coord_word_homo
        
        #Calculate reprojection error
        x1_cam1_rep, y1_cam1_rep = coord_cam1_rep[0:2, 0] / coord_cam1_rep[2, 0]
        x2_cam2_rep, y2_cam2_rep = coord_cam2_rep[0:2, 0] / coord_cam2_rep[2, 0]
        x3_cam3_rep, y3_cam3_rep = coord_cam3_rep[0:2, 0] / coord_cam3_rep[2, 0]

        err_reproject += (x1_cam1_rep-x1)**2 + (y1_cam1_rep-y1)**2 + (x2_cam2_rep-x2)**2 + (y2_cam2_rep-y2)**2 + (x3_cam3_rep-x3)**2 + (y3_cam3_rep-y3)**2
        
    P = np.stack(coord_word_list, axis=0)
    print(P.shape)
    
    return P, err_reproject

def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 300):

    p1s, p2s, p3s = pts1[:,:2], pts2[:,:2], pts3[:,:2]
    confidence1s, confidence2s, confidence3s = pts1[:,2], pts2[:,2], pts3[:,2]

    P, err = triangulate_3pts(C1, p1s, C2, p2s ,C3, p3s)
    P_12, _ = triangulate(C1, p1s, C2, p2s)
    P_23, _ = triangulate(C2, p2s, C3, p3s)
    P_31, _ = triangulate(C3, p3s, C1, p1s)
    
    # If only a point's confidence score is smaller than threshold, don't consider it and get P from triangulation using the other two points
    N_pts = len(pts1)
    for i in range(N_pts):
        if confidence1s[i] < Thres and confidence2s[i] > Thres and confidence3s[i] > Thres:
            P[i] = P_23[i]
            
        if confidence2s[i] < Thres and confidence3s[i] > Thres and confidence1s[i] > Thres:
            P[i] = P_31[i]
            
        if confidence3s[i] < Thres and confidence1s[i] > Thres and confidence2s[i] > Thres:
            P[i] = P_12[i]
    
    return P, err

'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''

def plot_3d_keypoint_video(pts_3d_video):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    for i in range(10):
        pts_word = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_word[index0,0], pts_word[index1,0]]
            yline = [pts_word[index0,1], pts_word[index1,1]]
            zline = [pts_word[index0,2], pts_word[index1,2]]
            ax.plot(xline, yline, zline, color = colors[j], alpha = 0.1 * i)
        np.set_printoptions(threshold = 1e6, suppress = True)    

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

#Extra Credit
if __name__ == "__main__":
         
    pts_3d_video = []
    # for loop in range(3):
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        C1 = K1.dot(M1)
        C2 = K2.dot(M2)
        C3 = K3.dot(M3)
      
        #Note - Press 'Escape' key to exit img preview and loop further 
        img = visualize_keypoints(im1, pts1)
        img = visualize_keypoints(im2, pts2)
        img = visualize_keypoints(im3, pts3)
        
        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 100)
        plot_3d_keypoint(P)
        print("Reprojection Error: {0}".format(err))
        pts_3d_video.append(P)

    plot_3d_keypoint_video(pts_3d_video)
    np.savez('q6_1.npz',pts_3d_video = pts_3d_video)