import os

if __name__ == '__main__':
    for i in [2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 1e-2]:
        command = "python\x20/sshare/home/yanzhiang/scripts/test_2_train_brain_mapping.py\x20--learning_rate\x20" + str(i)
        os.system(command)