import os

if __name__ == '__main__':
    for i in range(1, 11, 1):
        command = "python\x20all_extract.py\x20--ses\x20{}\x20".format(i)
        os.system(command)
        print(i, "finished")