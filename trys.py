# import argparse
#
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
#
# args = parser.parse_args()


import os
import numpy as np
root_path = 'D:\signal_data\G35_recorder_npydata'
file_path = os.listdir(root_path)
data_num = 1
print(file_path)



for idx in range(4, 11):
    data = np.load(os.path.join(root_path,file_path[idx]))
    label = np.load(os.path.join(root_path, file_path[idx + 7]), allow_pickle= True )

    fft_10 = data[data.files[0]]
    fft_80 = data[data.files[1]]
    RC_20 = data[data.files[2]]
    RC_40 = data[data.files[3]]

    label = label['labels']


    for i in range(RC_20.shape[0]):
        np.savez(os.path.join(root_path, 'train_data', str(data_num) + '.npz'), fft_10 = fft_10[i], fft_80 = fft_80[i], RC_20 = RC_20[i], RC_40 = RC_40[i])
        np.savez(os.path.join(root_path, 'train_label', str(data_num) + '.npz'), label = label[i])
        data_num += 1







