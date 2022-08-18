import numpy as np

np.random.seed(16)

def build_model(src,dst):
    '''
    fit the model with given points by solving the equations using svd
    '''
    n, c = src.shape
    M = np.zeros((2*n, (c+1)**2))

    M[0:n, 0:c] = src
    M[0:n, c] = 1
    M[0:n, -1] = -dst[:,0]
    M[0:n, -c-1:-1] = src
    M[0:n,-c-1:-1] *= -dst[:,0:1]
    
    M[n:2*n, c+1:2*c+1] = src
    M[n:2*n, 2*c+1] = 1
    M[n:2*n, -1] = -dst[:,1]
    M[n:2*n, -c-1:-1] = src
    M[n:2*n,-c-1:-1] *= -dst[:,1:2]
    

    _, _, Vh = np.linalg.svd(M)

    # check if matrix is degenerate (Not a full Rank matrix if last value in diagonal matrix Vh is 0)
    if np.isclose(Vh[-1, -1], 0):
        return None

    H = np.zeros((c+1, c+1))
    # solution (Null space) is the last column vector of the orthogonal matrix Vh
    H.flat[list(range((c+1)**2))] = Vh[-1, :] 
    H /= H[-1,-1]
    return H

def getTransformedPoints(H,pts):
    '''
    Transform source points so as to compare them with destination points to get the error between them
    '''
    src = np.c_[ pts, np.ones([pts.shape[0]]) ]
    dst = np.matmul(src,H.T)
    eps = 1e-20
    dst[dst[:, -1] == 0, -1] = eps
    dst /= dst[:, -1:]
    return dst[:,:-1]
    

def customRansac(srcPoints, destPoints, threshold=1.0, adaptive=False, error_type="Euclidean"):
    '''
    Custom Implementation of Ransac
    Input: 
        srcPoints: Points that are tranformed to destination points
        destPoints: Points that source points will be tranformed into
        threshold:  standard deviation of measurement errors
        adaptive: boolean variable True indicating run adaptive ransac mode
        error_type: distance measurement over which points are classified as inliers/outliers if they are less than t (t calculated below in code)
                    Currently 2 types supported namely Symmetric, Euclidean
    '''
    s = 4 # randomly picking s points
    w = 0.5 # probability a point is inlier (here prob inlier=1-outlier=0.5 initially given)
    n = srcPoints.shape[0] # number of points
    T = w*n 
    p = 0.99 
    N = int(np.ceil(np.log(1.0-p)/np.log(1.0-w**s))) # number of iterations
    best_consensus_set_size = 0
    best_inlier_error = np.inf
    best_inliers = None
    itr = 0
    dampCount = 0
    while True:
        itr += 1
        if itr > N:
            break

        indx = np.random.randint(0,n, size=s)
        s_src_pts, s_dst_pts = srcPoints[indx], destPoints[indx]

        # build model with these points
        H = build_model(s_src_pts, s_dst_pts)
        if H is None :
            continue

        # assuming measurement error is euclidean distance
        # the square of the distance (d**2) is the sum of squared x and y measurement errors.
        # taken root of distance
        if error_type.title() == "Euclidean":
            distance = np.sqrt(np.sum((getTransformedPoints(H,srcPoints) - destPoints)**2, axis=1))

        elif error_type.title() == "Symmetric":
            distance = np.sqrt(np.sum((getTransformedPoints(H,srcPoints) - destPoints)**2, axis=1) + \
                np.sum((srcPoints - getTransformedPoints(np.linalg.inv(H),destPoints))**2, axis=1))
        else:
            raise ValueError("Invalid error type for distance measurement in Custom-ransac. Only --error-type euclidean and symmetric are supported")

        t = np.sqrt(5.99)*threshold

        s_inliers = distance < t
        s_inlier_num = np.sum(s_inliers)

        # standard deviation of inliers
        s_inlier_error = np.std(distance[s_inliers])

        if s_inlier_num < T:

            # dampening T if initial assumption is pathologically incorrect
            T *=0.99
            dampCount += 1
            continue

        if  (s_inlier_num > best_consensus_set_size
        # in case of ties, pick consensus set with low standard deviation of inliers
                or (s_inlier_error < best_inlier_error
                    and s_inlier_num == best_consensus_set_size)
        ):
            best_consensus_set_size = s_inlier_num
            best_inlier_error = s_inlier_error
            best_inliers = s_inliers

            if adaptive:
                w_new = s_inlier_num/n
                if w_new > w:
                    w = w_new
                    if w == 1:
                        break
                    T = w*n
                    N = int(np.ceil(np.log(1.0-p)/np.log(1.0-w**s)))

    return best_inliers
