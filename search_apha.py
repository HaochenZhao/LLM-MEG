import numpy as np
import torch, mne
import multiprocessing as mp 
import time, os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.signal


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



def search_alpha_gpu(name, sub):
    alphas = [1e3, 1.5e3, 2e3, 2.5e3, 3e3, 3.5e3, 4e3, 4.5e3, 5e3, 5.5e3, 6e3, 6.5e3, 7e3, 7.5e3, 8e3, 8.5e3, 9e3, 9.5e3, 1e4, 1.5e4, 2e4, 2.5e4, 3e4, 3.5e4, 4e4, 4.5e4, 5e4, 5.5e4, 6e4, 6.5e4, 7e4, 7.5e4, 8e4, 8.5e4, 9e4, 9.5e4, 1e5]
    # Move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_path = "/data/zhiang/data/search_alpha/{}/sub-{:03d}/".format(name,sub)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for ses in range(1,11,1):
        X = torch.from_numpy(np.load('/data/zhiang/hidden_states/{}/sub-{:03d}/pca_200_hs_{}.npy'.format(name, sub, ses))).to(device).float()
        (aa, bb) = X.shape
        print(X.shape)
        train_size = int(aa*0.8)
        test_size = aa - train_size
        Xtrain = X[:train_size,:]
        Xtest = X[-test_size:,:]

        data_path = '/data/zhiang/holmes/preproc_new_upload/sub-{:03d}_ses-{:03d}_clean_raw'.format(sub, ses)
        epoch_fn = 'sub-{:03d}_ses-{:03d}_epoch_wdonset-epo.fif'.format(sub, ses)
        epochs = mne.read_epochs(os.path.join(data_path,epoch_fn))
        meg_data = epochs.get_data(picks=["mag"]) * 2e12
        meg_data = np.array(meg_data)

        if ses == 8 and sub == 2:
            meg_pad = np.zeros((1, 301))
            meg_data = np.insert(meg_data,102,meg_pad,axis=1)
            meg_data = np.insert(meg_data,104,meg_pad,axis=1)
            print(meg_data.shape)
        if ses == 8 and sub == 3:
            continue

        YY = torch.from_numpy(meg_data).to(device).float()
        YYtrain = YY[:train_size,:,50:150]
        YYtest = YY[-test_size:,:,50:150]

        As = np.zeros((269, 0))

        for alpha in alphas:
            CR = torch.zeros((269, 0), device=device)
            print(f"Processing alpha: {alpha}")

            for i in range(0, 100):
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
            CR = np.mean(CR, axis=1).reshape((269, 1))
            As = np.concatenate((As, CR), axis=1) 

            print("alpha:{} of session:{} GPU calculation complete".format(alpha, ses))
        print("session:{} all calculation complete".format(ses))
        print("As shape:", As.shape)
        bai = np.argmax(As.T, axis=0)
        best_alphas = []
        for index in bai:
            best_alphas.append(alphas[index])
        best_alphas = np.array(best_alphas)
        print(best_alphas)
        np.save(save_path+"ba_{}.npy".format(ses), best_alphas)

    


    
def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

if __name__ == '__main__':
    start_t = int(time.time())
    print("start at ", maket(start_t))

    name = "iter_0010000"
    
    for name in ["iter_0010000", "iter_0100000", "iter_0420000"]:
        for i in [1,2,3]:
            search_alpha_gpu(name, i)

    end_t = int(time.time())
    print("end at ", maket(end_t))

    used_t = maket(end_t-start_t)
    print("time used: ", used_t)