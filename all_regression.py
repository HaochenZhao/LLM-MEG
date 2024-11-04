import numpy as np
import torch
import multiprocessing as mp 
import time, os, mne, random
import matplotlib.pyplot as plt
from time import sleep

from wd_id import wd_id_ds

# Set code to GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.cuda.set_device(2)

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


def single(XX, YY_use, sub, alphas, type_):
    if type_ == 'down':
        time_points = 101
    else:
        time_points = 100

    d_nums = [0, 5120, 5121, 5124]
    d = d_nums[sub]
    CRs = torch.zeros((d, time_points, 0)).cuda()

    print("start moving X to GPU at: ", maket(int(time.time())))
    X_use = XX.cuda().float()
    print("finish moving X to GPU at: ", maket(int(time.time())))

    for alpha in alphas:
        print(f"Processing alpha: {alpha}")
        CRk = torch.zeros((d, time_points, 0)).cuda()
        clear()
        (aa, bb) = XX.shape
        print(XX.shape,YY.shape)    
        # 5-fold cross validation
        indices = torch.randperm(aa)
        fold_size = aa // 5

        for k in range(5):
            test_indices = indices[k*fold_size:(k+1)*fold_size] # test indices for current fold
            train_indices = torch.cat([indices[:k*fold_size], indices[(k+1)*fold_size:]]) # train indices for current fold
            
            Xtrain = X_use[train_indices,:]
            Xtest = X_use[test_indices,:]

            YYtrain = YY_use[train_indices,:,:]
            YYtest = YY_use[test_indices,:,:]

            CR = torch.zeros((d, 0)).cuda()
            print(f"Processing alpha: {alpha} k: {k}")

            for i in range(0, time_points):
                #print(f"Processing iteration {i} at: ", maket(int(time.time())))

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
                #print("finish iteration {} at: ".format(i), maket(int(time.time())))

            CRk = torch.cat((CRk, CR.unsqueeze(2)), dim=2)
        CRk = torch.mean(CRk, dim=2)
        CRs = torch.cat((CRs, CRk.reshape(d, time_points, 1)), dim=2)
    return CRs.cpu().numpy()

def loop(sub, ses, step, layer, YY_use, type_):
    # alphas for ridge regression
    alphas = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    name = "iter_{:07d}/layer{}".format(step*10000, layer)
    print("step: ", step, "layer: ", layer, "ses: ", ses, "sub: ", sub, "enter loop at: ", maket(int(time.time())))
    
    save_path = "/data/zhiang/data/as_dipole_regression/{}/".format(name)

    if not os.path.exists(save_path+"/sub-{:03d}".format(sub)):
        os.makedirs(save_path+"/sub-{:03d}".format(sub))
    X = np.load('/data/zhiang/hidden_states_ns/{}/pca200/pca_200_hs_{}.npy'.format(name, ses))
    
    clear() # empty cuda cache, seems no use
    X = torch.from_numpy(np.delete(X, wd_id_ds[sub][ses], axis=0)) # N words, 4096 features
    (aa, bb) = X.shape
    print(X.shape)
    
    print("start single computing at: ", maket(int(time.time())), "ses: ", ses, "sub: ", sub)
    CRs = single(X, YY_use, sub, alphas, type_)
    np.save(save_path + "sub-{:03d}/ses-{:03d}_{}.npy".format(sub, ses, type_), CRs)
    print("session:{} GPU calculation complete".format(ses))
    print("CRs shape:", CRs.shape)    

def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

def clear():
    torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    start_t = int(time.time())
    setup_seed(42)
    clear()
    print("start at ", maket(start_t))
    
    redo_list = []

    # step for model ckpt index, layer for model layer index
    sub = 3
    print("start sub_{:03d} ".format(sub))
    for ses in range(1,11,1):
        print("start session:{} at: ".format(ses), maket(int(time.time())))
        data_path = '/data/zhiang/data/extracted/all/{:03d}-{:03d}_wdonset_ico4_dSPM.npy'.format(sub, ses) # data path   
        meg_data = np.load(data_path) # meg data, N words, 5120 dipoles, 301 time points
        YY = torch.from_numpy(meg_data) # meg data, N words, 5120 dipoles, 301 time points

        print("start moving YY_up to GPU at: ", maket(int(time.time())))
        YY_use = YY[:,:,:100].cuda().float()
        print("finish moving YY_up to GPU at: ", maket(int(time.time())))

        for step in [20,25,30,35,40]:
            for layer in range(33):
                try:
                    if not os.path.exists("/data/zhiang/data/as_dipole_regression/iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'up')):
                        loop(sub, ses, step, layer, YY_use, 'up')
                    else:
                        print("skip up iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'up'))
                except Exception as e:
                    print("Error: ", e)
                    redo_list.append((sub, ses, step, layer, 'up'))

        print("start moving YY_middle to GPU at: ", maket(int(time.time())))
        YY_use = YY[:,:,100:200].cuda().float()
        print("finish moving YY_middle to GPU at: ", maket(int(time.time())))

        for step in [20,25,30,35,40]:
            for layer in range(33):
                try:
                    if not os.path.exists("/data/zhiang/data/as_dipole_regression/iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'middle')):
                        loop(sub, ses, step, layer, YY_use, 'middle')
                    else:
                        print("skip middle iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'middle'))
                except Exception as e:
                    print("Error: ", e)
                    redo_list.append((sub, ses, step, layer, 'middle'))

        print("start moving YY_down to GPU at: ", maket(int(time.time())))
        YY_use = YY[:,:,200:].cuda().float()
        print("finish moving YY_down to GPU at: ", maket(int(time.time())))

        for step in [20,25,30,35,40]:
            for layer in range(33):
                try:
                    if not os.path.exists("/data/zhiang/data/as_dipole_regression/iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'down')):
                        loop(sub, ses, step, layer, YY_use, 'down')
                    else:
                        print("skip down iter_{:07d}/layer{}/sub-{:03d}/ses-{:03d}_{}.npy".format(step*10000, layer, sub, ses, 'down'))
                except Exception as e:
                    print("Error: ", e)
                    redo_list.append((sub, ses, step, layer, 'down'))

                    
    end_t = int(time.time())
    print("end at ", maket(end_t))

    used_t = maket(end_t-start_t)
    print("time used: ", used_t)
    
    '''
    redo_2_list = []
    for redo in redo_list:
        try:
            loop(*redo)
        except Exception as e:
            print("Error: ", e)
            redo_2_list.append(redo)
    '''
    
    print("redo_list: ", redo_list)