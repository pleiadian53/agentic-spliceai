###############################################################################
'''This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.'''
###############################################################################

import h5py
import numpy as np
import sys
import time
from utils import *
from constants import *

start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']

h5f = h5py.File(data_dir + 'datafile'
                + '_' + sys.argv[1]
                + '.h5', 'r')

SEQ = h5f['SEQ'][:]
STRAND = h5f['STRAND'][:]
TX_START = h5f['TX_START'][:]
TX_END = h5f['TX_END'][:]
JN_START = h5f['JN_START'][:]
JN_END = h5f['JN_END'][:]
h5f.close()




# SEQ = SEQ.decode('utf-8')
# STRAND = STRAND.decode('utf-8')
# TX_START = TX_START.decode('utf-8')
# TX_END = TX_END.decode('utf-8')
# JN_START = JN_START.decode('utf-8')
# JN_END = JN_END.decode('utf-8')

h5f2 = h5py.File(data_dir + 'dataset'
                + '_' + sys.argv[1]
                + '.h5', 'w')

CHUNK_SIZE = 1

print("SEQ.shape[0]: ", SEQ.shape[0])
print("SEQ.shape[0]//CHUNK_SIZE: ", SEQ.shape[0]//CHUNK_SIZE)

for i in range(SEQ.shape[0]//CHUNK_SIZE):
    # Each dataset has CHUNK_SIZE genes
    
    if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
        NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0]%CHUNK_SIZE
    else:
        NEW_CHUNK_SIZE = CHUNK_SIZE

    print("NEW_CHUNK_SIZE: ", NEW_CHUNK_SIZE)

    X_batch = []
    Y_batch = [[] for t in range(1)]

    for j in range(NEW_CHUNK_SIZE):

        idx = i*CHUNK_SIZE + j




        SEQ_cvt = SEQ[idx].decode('utf-8')
        STRAND_cvt = STRAND[idx].decode('utf-8')
        TX_START_cvt = TX_START[idx].decode('utf-8')
        TX_END_cvt = TX_END[idx].decode('utf-8')
        JN_START_cvt = np.array([JN_START[idx][0].decode('utf-8')])
        JN_END_cvt = np.array([JN_END[idx][0].decode('utf-8')])
        # print("SEQ_cvt: ", SEQ_cvt)
        print("STRAND_cvt: ", STRAND_cvt)
        # print("TX_START_cvt: ", TX_START_cvt)
        # print("TX_END_cvt: ", TX_END_cvt)
        # print("JN_START_cvt: ", JN_START_cvt)
        # print("JN_END_cvt: ", JN_END_cvt)


        X, Y = create_datapoints(SEQ_cvt, STRAND_cvt,
                                 TX_START_cvt, TX_END_cvt,
                                 JN_START_cvt, JN_END_cvt)

        # print("X.shape: ", X.shape)
        # print("len(Y[0]): ", len(Y[0]))
        # print("\n\n")

        X_batch.extend(X)
        for t in range(1):
            Y_batch[t].extend(Y[t])

    X_batch = np.asarray(X_batch).astype('int8')
    print("X_batch.shape: ", X_batch.shape)
    for t in range(1):
        Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
    print("len(Y_batch[0]): ", len(Y_batch[0]))

    h5f2.create_dataset('X' + str(i), data=X_batch)
    h5f2.create_dataset('Y' + str(i), data=Y_batch)

h5f2.close()

print(("--- %s seconds ---" % (time.time() - start_time)))







# h5f2 = h5py.File(data_dir + 'dataset'
#                 + '_' + sys.argv[1]
#                 + '.h5', 'w')

# CHUNK_SIZE = 100

# for i in range(SEQ.shape[0]//CHUNK_SIZE):
#     # Each dataset has CHUNK_SIZE genes
    
#     if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
#         NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0]%CHUNK_SIZE
#     else:
#         NEW_CHUNK_SIZE = CHUNK_SIZE

#     X_batch = []
#     Y_batch = [[] for t in range(1)]

#     for j in range(NEW_CHUNK_SIZE):

#         idx = i*CHUNK_SIZE + j

#         X, Y = create_datapoints(SEQ_cvt, STRAND_cvt,
#                                  TX_START_cvt, TX_END_cvt,
#                                  JN_START_cvt, JN_END[idx])

#         X_batch.extend(X)
#         for t in range(1):
#             Y_batch[t].extend(Y[t])

#     X_batch = np.asarray(X_batch).astype('int8')
#     for t in range(1):
#         Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

#     h5f2.create_dataset('X' + str(i), data=X_batch)
#     h5f2.create_dataset('Y' + str(i), data=Y_batch)

# h5f2.close()

# print "--- %s seconds ---" % (time.time() - start_time)

###############################################################################         
