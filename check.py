import numpy as np
import torch
import multiprocessing as mp 
import time, os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.signal

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

def ridge_weights(X,Y,alpha):
    """
    X: Input data of shape (N, D) where N is the number of data points and D are features.
    Y: Target variable of shape (N, 1) 
    alpha: L2 regularization scalar value
    Return: weights of shape(D+1, 1) where D is features.
    """
    ones = np.ones((X.shape[0],1))  # (N,1)
    X_b = np.c_[ones,X]             # (N, D+1)
    lambda_I = alpha*np.identity(X_b.shape[1]) # (D+1, D+1) 

    weights = np.linalg.inv(X_b.T @ X_b + lambda_I) @ X_b.T @ Y # (D+1, D+1) @ (D+1, N) @ (N, 1) => (D+1, 1)
    return weights


def ridge_weights_gpu(X, Y, alpha):
    """
    GPU version of ridge_weights function using PyTorch.
    
    X: Input data tensor of shape (N, D) where N is the number of data points and D are features.
    Y: Target variable tensor of shape (N, 1)
    alpha: L2 regularization scalar value
    
    All input tensors are assumed to be on GPU.
    Return: weights tensor of shape (D+1, 1) on GPU.
    """
    N, D = X.shape
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_b = torch.cat([ones, X], dim=1)  # (N, D+1)
    
    lambda_I = alpha * torch.eye(D + 1, dtype=X.dtype, device=X.device)  # (D+1, D+1)
    
    # Compute (X_b.T @ X_b + lambda_I)
    XTX_plus_lambda = torch.matmul(X_b.T, X_b) + lambda_I
    
    # Compute inverse
    inv_matrix = torch.inverse(XTX_plus_lambda)
    
    # Compute weights
    weights = torch.matmul(torch.matmul(inv_matrix, X_b.T), Y)
    
    return weights



def predict(W,X):
    """
    W: Weights from ridge_weights function of shape(D+1, 1) where D is features.
    x: Input data point for prediction of shape (D, ) or in case you want to predict multiple points at once of shape (N, D)
    Return: y_hat - predicted value of shape (N, 1) if x was a 2d matrix else it will return shape (1, 1)
    """
    ones = np.ones((X.shape[0],1))  # (N,1)
    X_b = np.c_[ones,X]             # (N, D+1)
    y_hat = X_b @ W  # (N, D+1) @ (D+1, 1) => (N, 1)

    return y_hat

def predict_gpu(W, X):
    """
    GPU version of predict function using PyTorch.
    
    W: Weights tensor of shape (1, D+1) where D is the number of features.
    X: Input data tensor for prediction of shape (N, D) where N is the number of data points.
    Both W and X are assumed to be on GPU.
    
    Return: y_hat - predicted value tensor of shape (N, 1) on GPU.
    """
    N, D = X.shape
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_b = torch.cat([ones, X], dim=1)  # (N, D+1)
    y_hat = W @ X_b.T # (1, D+1) @ (D+1, N) => (1, N)

    return y_hat



def calc_pearson(X, Y): 
    XY = X*Y

    R = ((X*Y).mean(0) - X.mean(0)*Y.mean(0)) / (np.sqrt((X**2).mean(0) - (X.mean(0))**2) * np.sqrt((Y**2).mean(0) - (Y.mean(0))**2))

    return R

def calc_pearson_gpu(X, Y):
    """
    GPU version of calc_pearson function using PyTorch.
    
    X, Y: Input tensors of shape (N, D) where N is the number of samples and D is the number of features.
    Both X and Y are assumed to be on GPU.
    
    Returns: Pearson correlation coefficient tensor of shape (D,) on GPU.
    """
    # Ensure inputs are on GPU and have correct dtype
    X = X.float()
    Y = Y.float()
    
    # Calculate means
    X_mean = torch.mean(X, dim=0)
    Y_mean = torch.mean(Y, dim=0)
    
    # Calculate numerator
    numerator = torch.mean(X * Y, dim=0) - X_mean * Y_mean
    
    # Calculate denominator
    X_std = torch.sqrt(torch.mean(X**2, dim=0) - X_mean**2)
    Y_std = torch.sqrt(torch.mean(Y**2, dim=0) - Y_mean**2)
    denominator = X_std * Y_std
    
    # Calculate Pearson correlation coefficient
    R = numerator / denominator
    
    return R


def compute_cosine_similarity(A, B):
    M = A.shape[0]
    cosine_similarities = np.array([np.dot(A[i], B[i]) / (norm(A[i]) * norm(B[i])) for i in range(M)])
    return cosine_similarities

def compute_frobenius_norm(A, B):
    return norm(A - B)

def compute_mse(A, B):
    return np.mean((A - B) ** 2)

def run():

    '''
    CRs = np.load("/Users/mac/code/local/data/sCR001.npy")
    CRs = CRs.T
    CRs = CRs[:300, :]
    print(CRs.shape)

    results = np.load("/Users/mac/code/local/result/results.npy")
    #results = np.load("/Users/mac/code/local/data/sCR1.npy")
    #results = results.T
    #results = results[:300, :]
    print(results.shape)
    '''

    CRs = np.load("/Users/mac/code/local/ds/sCR_{}.npy".format(1e2))
    results = np.load("/Users/mac/code/local/ds/sCR_{}.npy".format(1e-2))

    print(compute_cosine_similarity(CRs, results))
    print(compute_frobenius_norm(CRs, results))
    print(compute_mse(CRs, results))

def sloop():
    alpha=0.01 # regularization scalar
    pool = mp.Pool(8)
    
    X = np.load('/data/zhiang/data/n/Pidden_states.npy')   # input data of shape (N, D) where N is the number of data points and D are features.
    Xtrain = X[:6873,:]
    Xtest = X[-1718:,:]
    YY = np.load('/data/zhiang/data/sub-001/meg_signals-001.npy')  # multiple Ys of shape (N, 1) each
    YYtrain = YY[:6873,:]
    YYtest = YY[-1718:,:]


    WW = np.zeros((269, 51, 0))
    PY = np.zeros((269, 1718, 0))
    CR = np.zeros((269, 0))

    for i in range(301):
        start_t = int(time.time())
        print("calculating {}th...  start at ".format(i), maket(start_t))

        Ys = YYtrain[:,:,i:i+1]
        (a,b,c) = Ys.shape
        Ys = Ys.reshape((a,b)).T
        Ys = Ys.reshape((b,a,1))

        Yt = YYtest[:,:,i:i+1]
        (a,b,c) = Yt.shape
        Yt = Yt.reshape((a,b)).T
        Yt = Yt.reshape((b,a,1))

        print(Xtrain.shape, Ys.shape)
        #exit(0)

        if not os.path.exists("/data/zhiang/data/checksensor/sWW.npy"):
            start_t = int(time.time())
            print("calculating {}th W... start at ".format(i), maket(start_t))
            results= [pool.apply(ridge_weights, args=(Xtrain,Y,alpha)) for Y in Ys] # parallelized computation of ridge weights for each Y
            #np.save("/data/zhiang/data/checksensor/Ws.npy", results)
            W_ = np.array(results)
            WW = np.concatenate((WW, W_), axis=2)
            end_t = int(time.time())
            print("{}th Ws end at ".format(i), maket(end_t))
        else:
            results = np.load("/data/zhiang/data/checksensor/sWW.npy")
            print("Ws has been calculated, shape:", results.shape)

        if not os.path.exists("/data/zhiang/data/checksensor/sPY.npy"):
            start_t = int(time.time())
            print("calculating {}th pred...  start at ".format(i), maket(start_t))
            predictions = [] 
            for W in results:  # assuming same order of data points as original Ys
                predictions.append(predict(W,Xtest))  # prediction using the computed weights   
            #np.save('/data/zhiang/data/checksensor/PY.npy', predictions) # save the nparray of predictions
            PYt = np.array(predictions)
            PY = np.concatenate((PY, PYt), axis=2)
            end_t = int(time.time())
            print("{}th pred end at ".format(i), maket(end_t))
        else:
            PYt = np.load("/data/zhiang/data/checksensor/sPY.npy")
            print("pred has been calculated, shape:", PYt.shape)

        if not os.path.exists("/data/zhiang/data/checksensor/sCR.npy"):
            start_t = int(time.time())
            print("calculating {}th R...  start at ".format(i), maket(start_t))
            Yt = Yt.reshape((b,a)).T
            PYt= PYt.reshape((b,a)).T
            corrs = calc_pearson(zs(Yt), zs(PYt))   
            #np.save('/data/zhiang/data/checksensor/CR.npy', corrs) # save the nparray of correlations
            R = np.array(corrs).reshape((269, 1))
            CR = np.concatenate((CR, R), axis=1)
            end_t = int(time.time())
            print("{}th R end at ".format(i), maket(end_t))
        else:
            corrs = np.load('/data/zhiang/data/checksensor/sCR.npy')
            print("R has been calculated, shape:", corrs.shape)

    print("calculation complete")
    print("WW shape:", WW.shape)
    print("PY shape:", PY.shape)
    print("CR shape:", CR.shape)

    np.save("/data/zhiang/data/checksensor/sWW.npy", WW)
    np.save("/data/zhiang/data/checksensor/sPY.npy", PY)
    np.save("/data/zhiang/data/checksensor/sCR.npy", CR)


def test_alpha():
    alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    #alphas = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]
    #alphas = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 7e7, 8e7, 9e7, 1e8]
    
    for alpha in alphas:
        print(alpha)
        pool = mp.Pool(32)
    
        X = np.load('/data/zhiang/hidden_states/iter_0420000/sub-001/pca_hs_1.npy')   # input data of shape (N, D) where N is the number of data points and D are features.
        Xtrain = X[:6873,:]
        Xtest = X[-1718:,:]
        YY = np.load('/data/zhiang/data/meg_signals.npy')  # multiple Ys of shape (N, 1) each
        h = 0
        YYtrain = YY[:6873,:,:]
        YYtest = YY[-1718:,:,:]


        WW = np.zeros((269, 51, 0))
        PY = np.zeros((269, 1718, 0))
        CR = np.zeros((269, 0))

        for i in range(50,150,1):
            start_t = int(time.time())
            print("calculating {}th...  start at ".format(i), maket(start_t))

            Ys = YYtrain[:,:,i:i+1]
            (a,b,c) = Ys.shape
            Ys = Ys.reshape((a,b)).T
            Ys = Ys.reshape((b,a,1))

            Yt = YYtest[:,:,i:i+1]
            (a,b,c) = Yt.shape
            Yt = Yt.reshape((a,b)).T
            Yt = Yt.reshape((b,a,1))

            print(Xtrain.shape, Ys.shape)
            #exit(0)

            start_t = int(time.time())
            print("calculating {}th W... start at ".format(i), maket(start_t))
            results= [pool.apply(ridge_weights, args=(Xtrain,Y,alpha)) for Y in Ys] # parallelized computation of ridge weights for each Y
            #np.save("/data/zhiang/data/checksensor/Ws.npy", results)
            W_ = np.array(results)
            WW = np.concatenate((WW, W_), axis=2)
            end_t = int(time.time())
            print("{}th Ws end at ".format(i), maket(end_t))

            start_t = int(time.time())
            print("calculating {}th pred...  start at ".format(i), maket(start_t))
            predictions = [] 
            for W in results:  # assuming same order of data points as original Ys
                predictions.append(predict(W,Xtest))  # prediction using the computed weights   
            #np.save('/data/zhiang/data/checksensor/PY.npy', predictions) # save the nparray of predictions
            PYt = np.array(predictions)
            PY = np.concatenate((PY, PYt), axis=2)
            end_t = int(time.time())
            print("{}th pred end at ".format(i), maket(end_t))


            start_t = int(time.time())
            print("calculating {}th R...  start at ".format(i), maket(start_t))
            Yt = Yt.reshape((b,a)).T
            PYt= PYt.reshape((b,a)).T
            corrs = calc_pearson(zs(Yt), zs(PYt))   
            #np.save('/data/zhiang/data/checksensor/CR.npy', corrs) # save the nparray of correlations
            R = np.array(corrs).reshape((269, 1))
            CR = np.concatenate((CR, R), axis=1)
            end_t = int(time.time())
            print("{}th R end at ".format(i), maket(end_t))

            
    np.save("/data/zhiang/data/check_alpha/sCR_{}.npy".format(alpha), CR)


def test_alpha_gpu():
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    #alphas = [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
    #alphas = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]
    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    
    X = torch.from_numpy(np.load('/data/zhiang/hidden_states/iter_0420000/sub-001/pca_hs_1.npy')).to(device).float()
    Xtrain = X[:6873,:]
    Xtest = X[-1718:,:]
    YY = torch.from_numpy(np.load('/data/zhiang/data/meg_signals.npy')).to(device).float()
    YYtrain = YY[:6873,:,:]
    YYtest = YY[-1718:,:,:]

    save_path = "/data/zhiang/data/check_alpha_4/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    for alpha in alphas:
        alpha = alpha * 10000
        WW = torch.zeros((269, 51, 0), device=device)
        PY = torch.zeros((269, 1718, 0), device=device)
        CR = torch.zeros((269, 0), device=device)
        print(f"Processing alpha: {alpha}")
        
        for i in range(50, 150):
            print(f"Processing iteration {i} at: ", maket(int(time.time())))
            
            Ys = YYtrain[:,:,i:i+1].permute(1, 0, 2)
            Yt = YYtest[:,:,i:i+1].permute(1, 0, 2)
            #print(Ys.shape, Yt.shape)

            # Ridge regression
            W = ridge_weights_gpu(Xtrain, Ys, alpha)
            #print(W.shape)
            W = W.permute(0, 2, 1).squeeze(1)
            #print(W.shape)

            # Prediction
            PYt = predict_gpu(W, Xtest)
            #print(PYt.shape)

            # Correlation
            Yt = Yt.squeeze(2).T
            PYt = PYt.T
            Yt_zscored = (Yt - Yt.mean(dim=0)) / Yt.std(dim=0)
            PYt_zscored = (PYt - PYt.mean(dim=0)) / PYt.std(dim=0)
            #print(Yt_zscored.shape, PYt_zscored.shape)
            R = calc_pearson_gpu(Yt_zscored, PYt_zscored)
            #print(R.shape)
            CR = torch.cat((CR, R.unsqueeze(1)), dim=1)
            print("finish iteration {} at: ".format(i), maket(int(time.time())))

        CR = CR.cpu().numpy()
        np.save(save_path + f"/sCR_{alpha}_gpu.npy", CR)

        print("alpha:{} GPU calculation complete".format(alpha))
        print("CR shape:", CR.shape)





def moving_average(i, window):
    w = np.ones(int(window)) / float(window)
    re = np.convolve(i, w, 'same')
    return re

def test_plot():
    #alphas = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    #alphas = [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
    #alphas = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]
    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    corrs = []
    
    save_path = "/data/zhiang/data/check_alpha_4"

    for alpha in alphas:
        print(alpha)
        sCR = np.load(save_path + "/sCR_{}_gpu.npy".format(alpha*10000))
        print(sCR.shape)

        results_data = np.mean(sCR, axis=1)
        #print(results_data.shape)
        results_data = np.mean(results_data, axis=0)
        #print(results_data.shape, results_data)
        corrs.append(results_data)

    corrs = np.array(corrs)
    title = "corr-alpha-50-150-(4-5)"
    color = "b"

    x = np.array(alphas)
    #x = np.log10(x)
    corrs_av = moving_average(corrs, 10)

    plt.title(title)
    plt.xlabel("alphas")
    plt.ylabel("corr.")
    plt.plot(x, corrs, color=color)
    plt.savefig(save_path + "/corr-alpha-50-150-(4-5).png")

def test_alpha_plot():
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    alphas_1 = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    alphas_2 = [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
    alphas_3 = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6]
    alphas_4 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

    X = torch.from_numpy(np.load('/data/zhiang/hidden_states/iter_0420000/sub-001/pca_hs_1.npy')).to(device).float()
    Xtrain = X[:6873,:]
    Xtest = X[-1718:,:]
    YY = torch.from_numpy(np.load('/data/zhiang/data/meg_signals.npy')).to(device).float()
    YYtrain = YY[:6873,:,:]
    YYtest = YY[-1718:,:,:]

    save_path_1 = "/data/zhiang/data/pca_check_alpha_1"
    save_path_2 = "/data/zhiang/data/pca_check_alpha_2"
    save_path_3 = "/data/zhiang/data/pca_check_alpha_3"
    save_path_4 = "/data/zhiang/data/pca_check_alpha_4"
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)
    if not os.path.exists(save_path_3):
        os.makedirs(save_path_3)
    if not os.path.exists(save_path_4):
        os.makedirs(save_path_4)

    for alpha in alphas_1:
        WW = torch.zeros((269, 51, 0), device=device)
        PY = torch.zeros((269, 1718, 0), device=device)
        CR = torch.zeros((269, 0), device=device)
        print(f"Processing alpha: {alpha}")
        
        for i in range(50, 150):
            print(f"Processing iteration {i} at: ", maket(int(time.time())))
            
            Ys = YYtrain[:,:,i:i+1].permute(1, 0, 2)
            Yt = YYtest[:,:,i:i+1].permute(1, 0, 2)
            #print(Ys.shape, Yt.shape)

            # Ridge regression
            W = ridge_weights_gpu(Xtrain, Ys, alpha)
            #print(W.shape)
            W = W.permute(0, 2, 1).squeeze(1)
            #print(W.shape)

            # Prediction
            PYt = predict_gpu(W, Xtest)
            #print(PYt.shape)

            # Correlation
            Yt = Yt.squeeze(2).T
            PYt = PYt.T
            Yt_zscored = (Yt - Yt.mean(dim=0)) / Yt.std(dim=0)
            PYt_zscored = (PYt - PYt.mean(dim=0)) / PYt.std(dim=0)
            #print(Yt_zscored.shape, PYt_zscored.shape)
            R = calc_pearson_gpu(Yt_zscored, PYt_zscored)
            #print(R.shape)
            CR = torch.cat((CR, R.unsqueeze(1)), dim=1)
            print("finish iteration {} at: ".format(i), maket(int(time.time())))

        CR = CR.cpu().numpy()
        np.save(save_path_1 + f"/sCR_{alpha}_gpu.npy", CR)

        print("alpha:{} GPU calculation complete".format(alpha))
        print("CR shape:", CR.shape)

    corrs = []
    for alpha in alphas_1:
        print(alpha)
        sCR = np.load(save_path_1 + "/sCR_{}_gpu.npy".format(alpha))
        print(sCR.shape)

        results_data = np.mean(sCR, axis=1)
        #print(results_data.shape)
        results_data = np.mean(results_data, axis=0)
        #print(results_data.shape, results_data)
        corrs.append(results_data)

    corrs = np.array(corrs)
    title = "corr-alpha-1"
    color = "b"

    x = np.array(alphas_1)
    x = np.log10(x)

    plt.title(title)
    plt.xlabel("alphas")
    plt.ylabel("corr.")
    plt.plot(x, corrs, color=color)
    plt.savefig(save_path_1 + "/corr-alpha-1.png")
    plt.close()


    for alpha in alphas_2:
        WW = torch.zeros((269, 51, 0), device=device)
        PY = torch.zeros((269, 1718, 0), device=device)
        CR = torch.zeros((269, 0), device=device)
        print(f"Processing alpha: {alpha}")
        
        for i in range(50, 150):
            print(f"Processing iteration {i} at: ", maket(int(time.time())))
            
            Ys = YYtrain[:,:,i:i+1].permute(1, 0, 2)
            Yt = YYtest[:,:,i:i+1].permute(1, 0, 2)
            #print(Ys.shape, Yt.shape)

            # Ridge regression
            W = ridge_weights_gpu(Xtrain, Ys, alpha)
            #print(W.shape)
            W = W.permute(0, 2, 1).squeeze(1)
            #print(W.shape)

            # Prediction
            PYt = predict_gpu(W, Xtest)
            #print(PYt.shape)

            # Correlation
            Yt = Yt.squeeze(2).T
            PYt = PYt.T
            Yt_zscored = (Yt - Yt.mean(dim=0)) / Yt.std(dim=0)
            PYt_zscored = (PYt - PYt.mean(dim=0)) / PYt.std(dim=0)
            #print(Yt_zscored.shape, PYt_zscored.shape)
            R = calc_pearson_gpu(Yt_zscored, PYt_zscored)
            #print(R.shape)
            CR = torch.cat((CR, R.unsqueeze(1)), dim=1)
            print("finish iteration {} at: ".format(i), maket(int(time.time())))

        CR = CR.cpu().numpy()
        np.save(save_path_2 + f"/sCR_{alpha}_gpu.npy", CR)

        print("alpha:{} GPU calculation complete".format(alpha))
        print("CR shape:", CR.shape)

    corrs = []
    for alpha in alphas_2:
        print(alpha)
        sCR = np.load(save_path_2 + "/sCR_{}_gpu.npy".format(alpha))
        print(sCR.shape)

        results_data = np.mean(sCR, axis=1)
        #print(results_data.shape)
        results_data = np.mean(results_data, axis=0)
        #print(results_data.shape, results_data)
        corrs.append(results_data)

    corrs = np.array(corrs)
    title = "corr-alpha-2"
    color = "b"

    x = np.array(alphas_2)
    x = np.log10(x)

    plt.title(title)
    plt.xlabel("alphas")
    plt.ylabel("corr.")
    plt.plot(x, corrs, color=color)
    plt.savefig(save_path_2 + "/corr-alpha-2.png")
    plt.close()

    for alpha in alphas_3:
        WW = torch.zeros((269, 51, 0), device=device)
        PY = torch.zeros((269, 1718, 0), device=device)
        CR = torch.zeros((269, 0), device=device)
        print(f"Processing alpha: {alpha}")
        
        for i in range(50, 150):
            print(f"Processing iteration {i} at: ", maket(int(time.time())))
            
            Ys = YYtrain[:,:,i:i+1].permute(1, 0, 2)
            Yt = YYtest[:,:,i:i+1].permute(1, 0, 2)
            #print(Ys.shape, Yt.shape)

            # Ridge regression
            W = ridge_weights_gpu(Xtrain, Ys, alpha)
            #print(W.shape)
            W = W.permute(0, 2, 1).squeeze(1)
            #print(W.shape)

            # Prediction
            PYt = predict_gpu(W, Xtest)
            #print(PYt.shape)

            # Correlation
            Yt = Yt.squeeze(2).T
            PYt = PYt.T
            Yt_zscored = (Yt - Yt.mean(dim=0)) / Yt.std(dim=0)
            PYt_zscored = (PYt - PYt.mean(dim=0)) / PYt.std(dim=0)
            #print(Yt_zscored.shape, PYt_zscored.shape)
            R = calc_pearson_gpu(Yt_zscored, PYt_zscored)
            #print(R.shape)
            CR = torch.cat((CR, R.unsqueeze(1)), dim=1)
            print("finish iteration {} at: ".format(i), maket(int(time.time())))

        CR = CR.cpu().numpy()
        np.save(save_path_3 + f"/sCR_{alpha}_gpu.npy", CR)

        print("alpha:{} GPU calculation complete".format(alpha))
        print("CR shape:", CR.shape)

    corrs = []
    for alpha in alphas_3:
        print(alpha)
        sCR = np.load(save_path_3 + "/sCR_{}_gpu.npy".format(alpha))
        print(sCR.shape)

        results_data = np.mean(sCR, axis=1)
        #print(results_data.shape)
        results_data = np.mean(results_data, axis=0)
        #print(results_data.shape, results_data)
        corrs.append(results_data)

    corrs = np.array(corrs)
    title = "corr-alpha-3"
    color = "b"

    x = np.array(alphas_3)
    x = np.log10(x)

    plt.title(title)
    plt.xlabel("alphas")
    plt.ylabel("corr.")
    plt.plot(x, corrs, color=color)
    plt.savefig(save_path_3 + "/corr-alpha-3.png")
    plt.close()


    for alpha in alphas_4:
        alpha = alpha * 10000
        WW = torch.zeros((269, 51, 0), device=device)
        PY = torch.zeros((269, 1718, 0), device=device)
        CR = torch.zeros((269, 0), device=device)
        print(f"Processing alpha: {alpha}")
        
        for i in range(50, 150):
            print(f"Processing iteration {i} at: ", maket(int(time.time())))
            
            Ys = YYtrain[:,:,i:i+1].permute(1, 0, 2)
            Yt = YYtest[:,:,i:i+1].permute(1, 0, 2)
            #print(Ys.shape, Yt.shape)

            # Ridge regression
            W = ridge_weights_gpu(Xtrain, Ys, alpha)
            #print(W.shape)
            W = W.permute(0, 2, 1).squeeze(1)
            #print(W.shape)

            # Prediction
            PYt = predict_gpu(W, Xtest)
            #print(PYt.shape)

            # Correlation
            Yt = Yt.squeeze(2).T
            PYt = PYt.T
            Yt_zscored = (Yt - Yt.mean(dim=0)) / Yt.std(dim=0)
            PYt_zscored = (PYt - PYt.mean(dim=0)) / PYt.std(dim=0)
            #print(Yt_zscored.shape, PYt_zscored.shape)
            R = calc_pearson_gpu(Yt_zscored, PYt_zscored)
            #print(R.shape)
            CR = torch.cat((CR, R.unsqueeze(1)), dim=1)
            print("finish iteration {} at: ".format(i), maket(int(time.time())))

        CR = CR.cpu().numpy()
        np.save(save_path_4 + f"/sCR_{alpha}_gpu.npy", CR)

        print("alpha:{} GPU calculation complete".format(alpha))
        print("CR shape:", CR.shape)

    corrs = []
    for alpha in alphas_4:
        print(alpha)
        sCR = np.load(save_path_4 + "/sCR_{}_gpu.npy".format(alpha*10000))
        print(sCR.shape)

        results_data = np.mean(sCR, axis=1)
        #print(results_data.shape)
        results_data = np.mean(results_data, axis=0)
        #print(results_data.shape, results_data)
        corrs.append(results_data)

    corrs = np.array(corrs)
    title = "corr-alpha-4"
    color = "b"

    x = np.array(alphas_4)
    #x = np.log10(x)

    plt.title(title)
    plt.xlabel("alphas")
    plt.ylabel("corr.")
    plt.plot(x, corrs, color=color)
    plt.savefig(save_path_4 + "/corr-alpha-4.png")
    plt.close()


    
def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

if __name__ == '__main__':
    start_t = int(time.time())
    print("start at ", maket(start_t))

    test_alpha_plot()

    end_t = int(time.time())
    print("end at ", maket(end_t))

    used_t = maket(end_t-start_t)
    print("time used: ", used_t)