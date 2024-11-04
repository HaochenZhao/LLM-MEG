import numpy as np
import torch, random
import multiprocessing as mp 
import time, os, mne
import matplotlib.pyplot as plt

from wd_id import wd_id_ds # use to drop words for different subjects


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


def loop(step, layer, ses, sub, YY):
    
    print("session:{} sub:{} step:{} layer:{} enter loop at: {}".format(ses, sub, step, layer, maket(int(time.time()))))
    # alphas for ridge regression
    alphas = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    name = "iter_{:07d}/layer{}".format(step*10000, layer)
    
    save_path = "/data/zhiang/data/as_sensor_regression/{}/".format(name)

    if not os.path.exists(save_path+"/sub-{:03d}".format(sub)):
        os.makedirs(save_path+"/sub-{:03d}".format(sub))
    X = np.load('/data/zhiang/hidden_states_ns/{}/pca200/pca_200_hs_{}.npy'.format(name, ses))

    clear() # empty cuda cache, seems no use
    CRs = torch.zeros((269, 301, 0), device=device) # 269 sensors, 301 time points, 0 alpha
    XX = torch.from_numpy(np.delete(X, wd_id_ds[sub][ses], axis=0)).to(device).float() # N words, 4096 features
    (aa, bb) = XX.shape
    print(XX.shape)
    
    # 5-fold cross validation
    indices = torch.randperm(aa)
    fold_size = aa // 5
    for alpha in alphas:
        CRk = torch.zeros((269, 301, 0), device=device) # 269 sensors, 301 time points, 0 fold
        for k in range(5):

            test_indices = indices[k*fold_size:(k+1)*fold_size] # test indices for current fold
            train_indices = torch.cat([indices[:k*fold_size], indices[(k+1)*fold_size:]]) # train indices for current fold

            Xtrain = XX[train_indices,:] # N*0.8 words, 4096 features
            Xtest = XX[test_indices,:] # N*0.2 words, 4096 features
            YYtrain = YY[train_indices,:,:] # N*0.8 words, 269 sensors, 301 time points
            YYtest = YY[test_indices,:,:] # N*0.2 words, 269 sensors, 301 time points
            #print(Xtrain.shape, Xtest.shape, YYtrain.shape, YYtest.shape)
    
            CR = torch.zeros((269, 0), device=device) # 269 sensors, 0 time points
            print(f"Processing alpha: {alpha}, k: {k}")
            for i in range(0, 301): # 301 time points loop
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

            CRk = torch.cat((CRk, CR.unsqueeze(2)), dim=2) # 269 sensors, 301 time points, up to 5 fold
        CRk = torch.mean(CRk, dim=2) # 269 sensors, 301 time points cross fold mean
        CRs = torch.cat((CRs, CRk.unsqueeze(2)), dim=2) # 269 sensors, 301 time points, up to 10 alpha
    CRs = CRs.cpu().numpy()
    np.save(save_path + "sub-{:03d}/ses-{:03d}.npy".format(sub, ses), CRs)

    print("session:{} sub:{} step:{} layer:{} GPU calculation complete at: {}".format(ses, sub, step, layer, maket(int(time.time()))))
    print("CRs shape:", CRs.shape)    


def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

def clear():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    global device
    clear()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_t = int(time.time())
    setup_seed(42)
    print("start at ", maket(start_t))

    for ses in range(1,11,1):
        for sub in range(1,4,1):
            data_path = '/data/zhiang/holmes/preproc_new_upload/sub-{:03d}_ses-{:03d}_clean_raw'.format(sub, ses) # data path   
            epoch_fn = 'sub-{:03d}_ses-{:03d}_epoch_wdonset-epo.fif'.format(sub, ses) # epoch file name
            epochs = mne.read_epochs(os.path.join(data_path,epoch_fn)) # read epochs
            meg_data = epochs.get_data(picks=["mag"]) * 2e12 # get meg data, scale by 2e12
            meg_data = np.array(meg_data)
            if ses == 8 and sub == 2:
                meg_pad = np.zeros((1, 301)) # pad for sensor 102 and 104
                meg_data = np.insert(meg_data,102,meg_pad,axis=1)
                meg_data = np.insert(meg_data,104,meg_pad,axis=1)
                print(meg_data.shape)
            if ses == 8 and sub == 3:
                continue
            
            YY = torch.from_numpy(meg_data).to(device).float() # meg data, N words, 269 sensors, 301 time points

            # step for model ckpt index, layer for model layer index
            for step in range(2,15,1):
                for layer in range(33):
                    loop(step, layer, ses, sub, YY)

    end_t = int(time.time())
    print("end at ", maket(end_t))

    used_t = maket(end_t-start_t)
    print("time used: ", used_t)