import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_layer(layer):
    save_path = '/home/zhiang/plot/most/layer_{}'.format(layer)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ii = list(range(301))
    tt = [(i-50)*4 for i in ii]
    x = np.array(tt)

    data_list = []
    data_list_p = []

    for step in [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]:
        data_d = np.zeros((301, 0))
        for sub in range(1, 4, 1):
            for ses in range(1, 7, 1):
                if sub == 3 and ses == 8:
                    continue
                if sub == 2 and ses == 8:
                    continue
                data_path = '/data/zhiang/pca_48_dipole/_iter_0{:02d}0000/_layer{}/00{}/_ses_{}_sensor.npy'.format(step, layer, sub, ses)
                data = np.load(data_path)
                
                data = np.max(data, axis=2) # max over 9 alphas
                
                data = np.mean(data, axis=0) # mean over 269 sensors
                #i=230
                #data = data[i:i+1, :].reshape((301, 1)) # only use the i-th sensor
                if np.isnan(data[0]):
                    print(step, sub, ses)
                    print(data)
                
                data_d = np.concatenate((data_d, data.reshape((301, 1))), axis=1)


        da = np.mean(data_d, axis=1) # mean over 3 subjects and 10 sessions
        plt.title('c, step:{}'.format(step*10000))
        plt.plot(x, da)
        plt.xlabel('time (ms)')
        plt.ylabel('correlation')
        plt.savefig(save_path+'/imgc_{}.png'.format(step*10000))
        plt.close()

        dap = da[50:150]
        da = np.mean(da, axis=0)
        data_list.append(da)
        dap = np.mean(dap, axis=0)
        data_list_p.append(dap)

    data_list = np.array(data_list)
    data_list_p = np.array(data_list_p)

    n = [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]
    plt.title('layer_{} c (000-400 ms)'.format(layer))
    plt.plot(n, data_list_p)
    plt.xlabel('step(*10000)')
    plt.ylabel('correlation')
    plt.savefig(save_path+'/c_000_400.png')
    plt.close()

    plt.title('layer_{} c (-200-1000 ms)'.format(layer))
    plt.plot(n, data_list)
    plt.xlabel('step(*10000)')
    plt.ylabel('correlation')
    plt.savefig(save_path+'/c_-200_1000.png')
    plt.close()


def plot_step(step):
    save_path = '/home/zhiang/plot/most/step_{}'.format(step)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ii = list(range(301))
    tt = [(i-50)*4 for i in ii]
    x = np.array(tt)

    data_list = []
    data_list_p = []

    for layer in [0,1,5,10,15,20,25,30]:
        data_d = np.zeros((301, 0))
        for sub in range(1, 4, 1):
            for ses in range(1, 7, 1):
                if sub == 3 and ses == 8:
                    continue
                if sub == 2 and ses == 8:
                    continue
                data_path = '/data/zhiang/pca_48_dipole/_iter_0{:02d}0000/_layer{}/00{}/_ses_{}_sensor.npy'.format(step, layer, sub, ses)
                data = np.load(data_path)
                
                data = np.max(data, axis=2) # max over 9 alphas
                
                data = np.mean(data, axis=0) # mean over 269 sensors
                #i=230
                #data = data[i:i+1, :].reshape((301, 1)) # only use the i-th sensor
                if np.isnan(data[0]):
                    print(step, sub, ses)
                    print(data)
                
                data_d = np.concatenate((data_d, data.reshape((301, 1))), axis=1)


        da = np.mean(data_d, axis=1) # mean over 3 subjects and 10 sessions
        plt.title('c, layer:{}'.format(layer))
        plt.plot(x, da)
        plt.xlabel('time (ms)')
        plt.ylabel('correlation')
        plt.savefig(save_path+'/imgc_{}.png'.format(layer))
        plt.close()

        dap = da[50:150]
        da = np.mean(da, axis=0)
        data_list.append(da)
        dap = np.mean(dap, axis=0)
        data_list_p.append(dap)

    data_list = np.array(data_list)
    data_list_p = np.array(data_list_p)

    n = [0,1,5,10,15,20,25,30]
    plt.title('step_0{:02d}0000 c (000-400 ms)'.format(step))
    plt.plot(n, data_list_p)
    plt.xlabel('layer')
    plt.ylabel('correlation')
    plt.savefig(save_path+'/c_000_400.png')
    plt.close()

    plt.title('step_0{:02d}0000 c (-200-1000 ms)'.format(step))
    plt.plot(n, data_list)
    plt.xlabel('layer')
    plt.ylabel('correlation')
    plt.savefig(save_path+'/c_-200_1000.png')
    plt.close()

def plot_3d_layer(layer):
    save_path = '/home/zhiang/plot/most3d/layer_{}'.format(layer)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ii = list(range(301))
    tt = [(i-50)*4 for i in ii]
    x = np.array(tt)
    ss = list(range(269))
    y = np.array(ss)

    data_list = []
    data_list_p = []

    for step in [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]:
        data_d = np.zeros((301, 269, 0))
        for sub in range(1, 4, 1):
            for ses in range(1, 7, 1):
                if sub == 3 and ses == 8:
                    continue
                if sub == 2 and ses == 8:
                    continue
                data_path = '/data/zhiang/pca_48_dipole/_iter_0{:02d}0000/_layer{}/00{}/_ses_{}_sensor.npy'.format(step, layer, sub, ses)
                data = np.load(data_path)
                
                data = np.max(data, axis=2) # max over 9 alphas
                
                #data = np.mean(data, axis=0) # mean over 269 sensors
                #i=230
                #data = data[i:i+1, :].reshape((301, 1)) # only use the i-th sensor
                #if np.isnan(data[0]):
                #    print(step, sub, ses)
                #    print(data)
                
                data_d = np.concatenate((data_d, data.reshape((301, 269, 1))), axis=2)


        da = np.mean(data_d, axis=2) # mean over 3 subjects and 10 sessions
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title('c, step:{}'.format(step*10000))
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, da.T)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('sensor')
        ax.set_zlabel('correlation')
        plt.tight_layout()  # Adjust layout to prevent clipping
        fig.savefig(save_path+'/imgc_{}.png'.format(step*10000))
        plt.close()

        dap = da[50:150]
        da = np.mean(da, axis=0)
        data_list.append(da)
        dap = np.mean(dap, axis=0)
        data_list_p.append(dap)

    data_list = np.array(data_list)
    data_list_p = np.array(data_list_p)

    n = [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('layer_{} c (000-400 ms)'.format(layer))
    X, Y = np.meshgrid(n, y)
    ax.plot_surface(X, Y, data_list_p.T)
    ax.set_xlabel('step(*10000)')
    ax.set_ylabel('sensor')
    ax.set_zlabel('correlation')
    fig.savefig(save_path+'/c_000_400.png')
    plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('layer_{} c (-200-1000 ms)'.format(layer))
    X, Y = np.meshgrid(n, y)
    ax.plot_surface(X, Y, data_list.T)
    ax.set_xlabel('step(*10000)')
    ax.set_ylabel('sensor')
    ax.set_zlabel('correlation')
    fig.savefig(save_path+'/c_-200_1000.png')
    plt.close()

def plot_3d_step(step):
    save_path = '/home/zhiang/plot/most3d/step_{}'.format(step)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ii = list(range(301))
    tt = [(i-50)*4 for i in ii]
    x = np.array(tt)
    ss = list(range(269))
    y = np.array(ss)

    data_list = []
    data_list_p = []

    for layer in [0,1,5,10,15,20,25,30]:
        data_d = np.zeros((301, 269, 0))
        for sub in range(1, 4, 1):
            for ses in range(1, 7, 1):
                if sub == 3 and ses == 8:
                    continue
                if sub == 2 and ses == 8:
                    continue
                data_path = '/data/zhiang/pca_48_dipole/_iter_0{:02d}0000/_layer{}/00{}/_ses_{}_sensor.npy'.format(step, layer, sub, ses)
                data = np.load(data_path)
                
                data = np.max(data, axis=2) # max over 9 alphas
                
                #data = np.mean(data, axis=0) # mean over 269 sensors
                #i=230
                #data = data[i:i+1, :].reshape((301, 1)) # only use the i-th sensor
                #if np.isnan(data[0]):
                #    print(step, sub, ses)
                #    print(data)
                
                data_d = np.concatenate((data_d, data.reshape((301, 269, 1))), axis=2)


        da = np.mean(data_d, axis=2) # mean over 3 subjects and 10 sessions
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title('c, layer:{}'.format(layer))
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, da.T)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('sensor')
        ax.set_zlabel('correlation')
        plt.tight_layout()  # Adjust layout to prevent clipping
        fig.savefig(save_path+'/imgc_{}.png'.format(layer))
        plt.close()

        dap = da[50:150]
        da = np.mean(da, axis=0)
        data_list.append(da)
        dap = np.mean(dap, axis=0)
        data_list_p.append(dap)

    data_list = np.array(data_list)
    data_list_p = np.array(data_list_p)

    n = [0,1,5,10,15,20,25,30]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('step_0{:02d}0000 c (000-400 ms)'.format(step))
    X, Y = np.meshgrid(n, y)
    ax.plot_surface(X, Y, data_list_p.T)
    ax.set_xlabel('layer')
    ax.set_ylabel('sensor')
    ax.set_zlabel('correlation')
    fig.savefig(save_path+'/c_000_400.png')
    plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title('step_0{:02d}0000 c (-200-1000 ms)'.format(step))
    X, Y = np.meshgrid(n, y)
    ax.plot_surface(X, Y, data_list.T)
    ax.set_xlabel('layer')
    ax.set_ylabel('sensor')
    ax.set_zlabel('correlation')
    fig.savefig(save_path+'/c_-200_1000.png')
    plt.close()


if __name__ == "__main__":
    '''
    for layer in [0,1,5,10,15,20,25,30]:
        plot_layer(layer)
    '''
    '''
    for step in [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]:
        plot_step(step)
    '''
    
    for layer in [0,1,5,10,15,20,25,30]:
        plot_3d_layer(layer)
    
    for step in [1,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25,27,28,30,32,33,35,37,38,40,42]:
        plot_3d_step(step)
    
