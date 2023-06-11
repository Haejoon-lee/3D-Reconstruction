import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def KernelResponse(im, x, y, kxs, kys):    
    H, W = im.shape[0:2]
    xs, ys = kxs + x, kys + y
    xs, ys = np.clip(xs, 0, W-1).astype(np.int32), np.clip(ys, 0, H-1).astype(np.int32)
    response = im[ys, xs, :]
    return response

def epipolarCorrespondence(im1, im2, F, x1, y1):

    coord_img1_homo = np.array([[x1], [y1], [1]], dtype=np.float32)
    # Calculate epipolar line on im2
    epipolarline_img2 = F @ coord_img1_homo

    # Set search boundary
    r = 100
    ll = np.array([1, 0, -(x1-r)], dtype=np.float32).reshape(-1, 1)
    lt = np.array([0, 1, -(y1-r)], dtype=np.float32).reshape(-1, 1)
    lr = np.array([1, 0, -(x1+r)], dtype=np.float32).reshape(-1, 1)
    lb = np.array([0, 1, -(y1+r)], dtype=np.float32).reshape(-1, 1)
    lines_boundary = [ll, lt, lr, lb]
    
    # Get intersected points between search boundary and epipolar line
    points_intersect = [np.cross(l.reshape(-1), epipolarline_img2.reshape(-1)).reshape(-1, 1) for l in lines_boundary]

    points_end = []
    for points in points_intersect:
        if np.abs(points[2, 0]) > 0.0000001:
            points_homo = points / points[2, 0]
            dist_max = np.max(np.abs(points_homo[0:2, :]-coord_img1_homo[0:2, :]))
            if dist_max < r*(1.1):
                points_end.append(points_homo)

    search_begin = None
    search_end = None
    if len(points_end) > 2:
        point0_end = points_end[0]
        for i in range(1, len(points_end)):
            coord_img1_homo = points_end[i]
            if np.min(np.abs(point0_end[0:2, :]-coord_img1_homo[0:2, :])) > r*0.1:
                search_begin = point0_end[0:2, :]
                search_end = coord_img1_homo[0:2, :]
                break
    else:
        search_begin = points_end[0][0:2, :]
        search_end = points_end[1][0:2, :]
    
    # Generate Gaussian weighting kernel
    kernel_r = 20
    kernel_xaxis = np.arange(-kernel_r, kernel_r+1, 1.0)
    kernel_yaxis = np.arange(-kernel_r, kernel_r+1, 1.0)
    kernel_xaxis, kernel_yaxis = np.meshgrid(kernel_xaxis, kernel_yaxis)
    kernel_xaxis, kernel_yaxis = kernel_xaxis.reshape(-1), kernel_yaxis.reshape(-1)

    STD = kernel_r / 2.0
    kernel_window = np.exp( -( (kernel_xaxis**2 + kernel_yaxis**2)/(2*STD**2) ) ) / (2*np.pi*STD**2)

    response_img1 = KernelResponse(im1, x1, y1, kernel_xaxis, kernel_yaxis)
    
    #Calculate distances by compare kernel response and get the best matching point on im2
    dist_min = np.Inf
    num_steps = int(np.sum((search_begin-search_end)**2)**0.5)
    x_step = (search_end[0, 0] - search_begin[0, 0]) / num_steps
    y_step = (search_end[1, 0] - search_begin[1, 0]) / num_steps
    x2, y2 = search_begin[0, 0], search_begin[1, 0]
    x2_best, y2_best = -1, -1
    for i in range(num_steps):
        response_img2 = KernelResponse(im2, x2, y2, kernel_xaxis, kernel_yaxis)
        dist = np.sum((response_img2 - response_img1)**2, axis=1)**0.5 # (Nk,)
        dist = np.sum(dist*kernel_window)
        
        if dist < dist_min:
            dist_min = dist
            x2_best, y2_best = x2, y2

        x2, y2 = x2 + x_step, y2 + y_step
    
    return x2_best, y2_best

if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)
    
    epipolarMatchGUI(im1, im2, F)
    
    np.savez('q4_1.npz', F, pts1, pts2)

