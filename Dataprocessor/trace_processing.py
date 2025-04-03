import pandas as pd
import numpy as np
import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#base_path = os.path.dirname(os.path.abspath(__file__))
#data_path = os.path.join(base_path, '../input_data')
data_path = 'E:\DSL\Modeling/Integrated/rawdata'

def argparse():
    parser = ArgumentParser()
    parser.add_argument('--file_name', action = 'store')
    return parser.parse_args()

import os
import numpy as np
import pandas as pd

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def trace_generator(file_name, data_path, tgt_path):
    """
    CSV 파일에서 'Timestamp', 'Smoothed_X', 'Smoothed_Y' 데이터를 추출하여
    timestamp와 X, Y 값을 플롯한 그래프를 생성합니다.
    그래프는 오렌지색 선(시각 vs X)과 파란색 선(시각 vs Y)을 겹쳐서 그립니다.
    생성된 그래프는 tgt_path 폴더에 "{파일명}_trace.png"로 저장되며,
    저장된 이미지 파일의 전체 경로를 반환합니다.
    
    :param file_name: CSV 파일 이름 (예: "sample.csv")
    :param data_path: CSV 파일들이 저장된 폴더 경로 (예: "./rawdata")
    :param tgt_path: 생성된 이미지 파일을 저장할 폴더 경로 (예: "./inputdata/trace")
    :return: 생성된 이미지 파일의 전체 경로 또는 None (오류 발생 시)
    """
    full_path = os.path.join(data_path, file_name)
    try:
        df = pd.read_csv(full_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"{file_name} 파일 읽기 오류: {e}")
        return None

    # 필수 컬럼 확인 (대소문자에 주의)
    required_cols = ['timestamp', 'smooth_x', 'smooth_y']
    for col in required_cols:
        if col not in df.columns:
            print(f"{file_name} 파일에 '{col}' 컬럼이 없습니다.")
            return None

    try:
        # 첫 번째 타임스탬프를 0으로 맞춤
        time_vals = np.array(df['timestamp'])
        time_vals = time_vals - time_vals[0]
        X = np.array(df['smooth_x'])
        Y = np.array(df['smooth_y'])
    except Exception as e:
        print(f"{file_name} 데이터 처리 오류: {e}")
        return None

    #print('dataset successfully loaded')

    plt.figure(figsize=(10, 6))
    # timestamp와 X, Y를 각각 플롯
    plt.plot(time_vals, X, color="orange")
    plt.plot(time_vals, Y, color="blue")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.axis('off')
    #print('plot successfully generated')

    # tgt_path 폴더가 없으면 생성
    os.makedirs(tgt_path, exist_ok=True)
    name, _ = os.path.splitext(file_name)
    output_file = os.path.join(tgt_path, f"{name}_trace.png")
    try:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        print(f"그래프 저장 완료: {output_file}")
    except Exception as e:
        print(f"{file_name} 이미지 저장 오류: {e}")
        plt.close()
        return None
    finally:
        plt.close()
    
    return output_file



def main():
    parser = argparse()
    file_name = parser.file_name
    trace_generator(file_name)
    

if __name__ == '__main__':
    main()
    print('done generating eye trace')