import scipy
import numpy as np
from io import StringIO
from scipy.io import mmread
import argparse

def save_file(filename, arr):
    f = open(filename, "wb")
    f.write(len(arr).to_bytes(8, 'little'))
    f.write(int(0).to_bytes(8, 'little'))
    f.write(arr.tobytes())
    f.close()

def trans(input, output):
    mtx = mmread(input)
    print("read done")
    output_path = output
    
    mtx = mtx.tocsr()
    print("to csr done")
    col = mtx.indptr.astype(np.int64)
    save_file(output_path + ".col", col)
    del col
    dst = mtx.indices.astype(np.int64)  
    save_file(output_path + ".dst", dst)
    del dst
    val = mtx.data.astype(np.float32)
    save_file(output_path + ".val", val)
    del val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert matrix market format to binary format')
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('output', type=str, help='output file')
    args = parser.parse_args()
    trans(args.input, args.output)
