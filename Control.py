import subprocess
import os
import threading
import queue
import json
import time
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend 사용
import matplotlib.pyplot as plt

# 실행할 스크립트들이 위치한 폴더 경로
data_generation = './tracking_api'
data_processing = './dataprocessor'
cal_dir = './Calibration'
save_dir = './rawdata'
data_dir = ''

sys.path.append(os.path.abspath(data_generation))
from detection import detection_processor  # 원래의 detection_processor 함수가 있으나, 아래에서 확장한 함수를 사용할 예정입니다.

def dirempty(dir):
    return not os.listdir(dir)

# tracking.py에서 출력되는 데이터를 저장할 Queue
tracking_queue = queue.Queue(maxsize=1000)
# detection 결과를 저장할 Queue
detection_queue = queue.Queue(maxsize=1000)

# 전역 누적 데이터와 스레드 안전을 위한 Lock
plot_data = []
plot_data_lock = threading.Lock()

def read_tracking_stderr(proc):
    """realtime_tracking.py의 stderr에서 출력되는 내용을 실시간으로 읽어 출력합니다."""
    while True:
        err_line = proc.stderr.readline()
        if not err_line:
            break
        print("STDERR:", err_line.strip())

def read_tracking_data(proc, tracking_queue):
    """stdout에서 한 줄씩 읽어 JSON 파싱 후 tracking_queue에 저장합니다."""
    while True:
        try:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.strip())
                tracking_queue.put(data)
            except json.JSONDecodeError:
                print("JSON 파싱 오류:", line)
        except Exception as e:
            print("read_tracking_data 예외 발생:", e)
            break

def detection_processor_with_plot(tracking_queue, detection_queue, fixation_threshold=50, regression_threshold=100, y_tolerance=10):
    """
    tracking_queue에서 데이터를 꺼내어 간단한 판정을 내고 detection_queue에 저장합니다.
    동시에 전역 plot_data 리스트에도 결과를 추가하여 플롯에 활용할 수 있도록 합니다.
    """
    prev_event = None
    while True:
        try:
            event = tracking_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if prev_event is None:
            label = "No previous event"
        else:
            delta_x = event['screen_x'] - prev_event['screen_x']
            delta_y = abs(event['screen_y'] - prev_event['screen_y'])
            if delta_y > y_tolerance:
                label = "New Line"
            elif delta_x < -regression_threshold:
                label = "Regression"
            elif abs(delta_x) <= fixation_threshold:
                label = "Fixation"
            else:
                label = "Saccade"
        
        detection_result = {
            "timestamp": event["timestamp"],
            "label": label,
            "event": event
        }
        detection_queue.put(detection_result)
        with plot_data_lock:
            plot_data.append(detection_result)
        prev_event = event

def save_detection_plot(image_dir):
    """
    전역 plot_data에 누적된 detection 결과를 이용해 플롯을 그리고,
    지정한 디렉토리에 "fixmap.png" 파일로 저장합니다.
    주기적으로 (예: 5초마다) 실행됩니다.
    """
    # image_dir가 없으면 생성
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    while True:
        # 5초마다 플롯 업데이트
        time.sleep(5)
        with plot_data_lock:
            data_copy = list(plot_data)
        xs, ys, colors = [], [], []
        # label 별 색상 매핑
        label_colors = {
            "No previous event": "gray",
            "New Line": "blue",
            "Regression": "red",
            "Fixation": "green",
            "Saccade": "orange"
        }
        for det in data_copy:
            event = det["event"]
            x = event["screen_x"]
            y = event["screen_y"]
            label = det["label"]
            xs.append(x)
            ys.append(y)
            colors.append(label_colors.get(label, "black"))
        
        # 플롯 생성
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, 1920)
        ax.set_ylim(0, 1080)
        ax.invert_yaxis()  # 화면 좌표계와 맞추기 위해 y축 반전
        ax.set_xlabel("Screen X")
        ax.set_ylabel("Screen Y")
        ax.set_title("Detection Results")
        ax.scatter(xs, ys, c=colors, s=20)
        
        # 플롯 결과를 파일로 저장 (매번 같은 이름으로 덮어쓰기)
        filename = os.path.join(image_dir, "fixmap.png")
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved plot to {filename}")

def main():
    if dirempty(cal_dir):
        print("Calibration directory is empty. Running calibration script...")
        calibration_cmd = ["python", "eye_calibration.py", "--data_dir", cal_dir]
        subprocess.run(calibration_cmd)

    print('end of calibration verification')
    realtime_tracking_path = os.path.join(data_generation, "realtime_tracking.py")
    proc = subprocess.Popen(
        ["python", realtime_tracking_path, "--data_dir", save_dir, "--cal_dir", cal_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print('started tracking your eyes')
    
    reader_thread = threading.Thread(target=read_tracking_data, args=(proc, tracking_queue), daemon=True)
    reader_thread.start()
    
    err_thread = threading.Thread(target=read_tracking_stderr, args=(proc,), daemon=True)
    err_thread.start()
    
    detector_thread = threading.Thread(target=detection_processor_with_plot, args=(tracking_queue, detection_queue), daemon=True)
    detector_thread.start()
    print('started detecting the information')
    
    # 플롯 결과를 저장하는 스레드 시작
    image_dir = './inputdata/fixmap'
    plot_thread = threading.Thread(target=save_detection_plot, args=(image_dir,), daemon=True)
    plot_thread.start()
    
    # 메인 루프: detection 결과를 콘솔에 출력 (원하는 경우)
    try:
        while True:
            try:
                result = detection_queue.get(timeout=1)
                print("Detection result:", result)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("control.py 종료 요청됨.")
    finally:
        proc.terminate()
        reader_thread.join()
        detector_thread.join()
        err_thread.join()
        plot_thread.join()

if __name__ == '__main__':
    main()
