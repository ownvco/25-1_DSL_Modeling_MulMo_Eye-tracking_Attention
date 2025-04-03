import pandas as pd
import numpy as np
import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../rawdata')
tgt_path = os.path.join(base_path, '../inputdata/trace')

def argparse():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', action = 'store')
    parser.add_argument('--tgt_dir', action = 'store')
    return parser.parse_args()

def trace_generator(file_name):
    df = pd.read_csv(os.path.join(data_path, file_name), encoding = 'utf-8-sig')
    
    time = np.array(df['Timestamp'])
    time = time - time[0]   # 첫 번째 timestep = 0초로 설정!
    
    X = np.array(df['Smoothed_X'])
    Y = np.array(df['Smoothed_Y'])

    print('dataset successfully loaded')

    plt.figure(figsize=(10, 6))

    # LX와 LY를 각각 시간에 따라 플롯 (오렌지색, 파란색)
    plt.plot(time, X, color="orange")
    plt.plot(time, Y, color="blue")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.axis('off')

    print('plot successfully generated')

    # 그래프를 graph.png 파일로 저장 (jpg로 저장하려면 확장자만 바꾸면 됩니다)
    name, _ = os.path.splitext(file_name)
    plt.savefig(os.path.join(tgt_path,f"{name}_trace.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

# folder으로 데이터 세부 디렉토리 설정정
def multiple_generator(folder):
    data_dir = os.path.join(data_path, folder)
    for dirs in os.listdir(data_dir):
        trace_generator(dirs)


def main():
    multiple_generator(data_path)
    

if __name__ == '__main__':
    main()
    print('done generating eye trace')