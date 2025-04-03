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
import matplotlib.patches as patches
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 실행할 스크립트들이 위치한 폴더 경로
data_generation = './new_tracking_api'
data_processing = './dataprocessor'
cal_dir = './Calibration_new'
save_dir = './rawdata'         # CSV 파일들이 저장되는 폴더
ocr_dir = './OCR'
screenshot_dir = './screenshots'
classifier_dir = './classifier'
feedback_dir = './LLM'

# classifier가 사용할 이미지 경로 (예: DDD_efficientnet.py의 입력 경로와 일치)
classifier_trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputdata')

sys.path.append(os.path.abspath(data_generation))
sys.path.append(os.path.abspath(data_processing))
sys.path.append(os.path.abspath(ocr_dir))
sys.path.append(os.path.abspath(classifier_dir))
sys.path.append(os.path.abspath(feedback_dir))

from detection import detection_processor  # 수정된 detection_processor (stop_event 인자 추가)
from processor import fixation_group_processor  # 수정된 fixation_group_processor (stop_event 인자 추가)
from dummy import capture_screenshots_worker
from DDD_efficientnet import classifier
# trace_generator: CSV 파일에서 데이터를 읽어 플롯을 생성한 후 이미지를 저장하고, 이미지 파일의 경로를 반환하는 함수.
from trace_processing import trace_generator
from run_paddleocr import process_screenshot_queue
from feedback import generate_feedback

# 전역 종료 이벤트 생성
stop_event = threading.Event()

# Global queues
tracking_queue = queue.Queue(maxsize=5000)
detection_queue = queue.Queue(maxsize=5000)
fixation_queue = queue.Queue(maxsize=5000)
regression_queue = queue.Queue(maxsize=5000)
fixation_plot_queue = queue.Queue(maxsize=5000)
regression_plot_queue = queue.Queue(maxsize=5000)
screenshot_queue = queue.Queue(maxsize=5000)
classification_queue = queue.Queue(maxsize=5000)
classifier_input_queue = queue.Queue(maxsize=5000)
# OCR 결과를 받을 큐 (OCR 결과에는 label과 text가 포함됨)
ocr_queue = queue.Queue(maxsize=5000)

# 기존 큐의 put 메서드 오버라이드 (추가 처리를 위해)
original_detection_put = detection_queue.put
def custom_detection_put(item, block=True, timeout=None):
    original_detection_put(item, block, timeout)
    if item.get("label") == "Regression":
        regression_queue.put(item, block, timeout)
        regression_plot_queue.put(item, block, timeout)
detection_queue.put = custom_detection_put

original_fixation_put = fixation_queue.put
def custom_fixation_put(item, block=True, timeout=None):
    original_fixation_put(item, block, timeout)
    fixation_plot_queue.put(item, block, timeout)
fixation_queue.put = custom_fixation_put

def read_tracking_stderr(proc, stop_event):
    while not stop_event.is_set():
        try:
            err_line = proc.stderr.readline()
            if not err_line:
                break
            print("STDERR:", err_line.strip())
        except Exception as e:
            print("read_tracking_stderr 예외 발생:", e)
            stop_event.set()
            break

def read_tracking_data(proc, tracking_queue, stop_event):
    while not stop_event.is_set():
        try:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.strip())
                tracking_queue.put(data)
            except json.JSONDecodeError:
                pass
                #print("JSON 파싱 오류:", line)
        except Exception as e:
            print("read_tracking_data 예외 발생:", e)
            stop_event.set()
            break

def save_output_plot(reg_plot_queue, fix_plot_queue, update_interval, output_path, stop_event):
    while not stop_event.is_set():
        try:
            reg_data = []
            fix_data = []
            try:
                while True:
                    reg_item = reg_plot_queue.get_nowait()
                    reg_data.append(reg_item)
            except queue.Empty:
                pass
            try:
                while True:
                    fix_item = fix_plot_queue.get_nowait()
                    fix_data.append(fix_item)
            except queue.Empty:
                pass

            if reg_data or fix_data:
                plt.figure(figsize=(8, 6))
                if reg_data:
                    x_reg = [item["event"]["smooth_x"] for item in reg_data]
                    y_reg = [item["event"]["smooth_y"] for item in reg_data]
                    plt.scatter(x_reg, y_reg, color="red", label="Regression")
                ax = plt.gca()
                for item in fix_data:
                    center = item["center"]
                    width = item["width"]
                    height = item["height"]
                    lower_left = (center[0] - width/2, center[1] - height/2)
                    rect = patches.Rectangle(lower_left, width, height, linewidth=1, edgecolor="blue", facecolor="none")
                    ax.add_patch(rect)
                plt.legend()
                plt.title("Regression and Fixation Groups")
                plt.xlabel("Smooth X")
                plt.ylabel("Smooth Y")
                plt.savefig(output_path)
                plt.close()
            time.sleep(update_interval)
        except Exception as e:
            print("save_output_plot 예외 발생:", e)
            stop_event.set()
            break

def save_screenshot(screenshot_queue, regression_path, fixation_path, stop_event):
    reg_cnt = 1
    fix_cnt = 1
    while not stop_event.is_set():
        try:
            if not screenshot_queue.empty():
                obj = screenshot_queue.get(timeout=1)
                typ = obj['type']
                img = obj['screenshot']

                if typ == 'regression':
                    file_name = f'regression_{reg_cnt}.png'
                    img.save(os.path.join(regression_path, file_name))
                    reg_cnt += 1
                elif typ == 'fixation':
                    file_name = f'fixation_{fix_cnt}.png'
                    img.save(os.path.join(fixation_path, file_name))
                    fix_cnt += 1
        except queue.Empty:
            continue
        except Exception as e:
            print("screenshot saving worker error:", e)
            time.sleep(1)

# classifier 스레드: classifier_input_queue에서 이미지 파일 경로를 받아 classifier 함수를 호출
def run_classifier(classifier_input_queue, classification_queue, stop_event):
    while not stop_event.is_set():
        try:
            image_file = classifier_input_queue.get(timeout=1)
            #print("Classifier processing image:", image_file)
            start_time = time.time()
            result, inference_time = classifier(image_file)
            classification_queue.put(result)
            print("Classifier result:", result)
            end_time = time.time()
            print(end_time - start_time)
        except queue.Empty:
            continue
        except Exception as e:
            print("Classifier error:", e)
            continue

def logging_regression(stop_event):
    while not stop_event.is_set():
        try:
            reg_item = regression_queue.get(timeout=1)
            #print("Regression event captured:", reg_item)
        except queue.Empty:
            continue
        except Exception as e:
            print("logging_regression 예외 발생:", e)
            stop_event.set()
            break

# --- CSV 파일 생성 감지를 위한 watchdog 관련 코드 ---
class CSVCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        # 파일이 디렉터리가 아니고 CSV 파일인 경우
        if not event.is_directory and event.src_path.endswith('.csv'):
            file_name = os.path.basename(event.src_path)
            #print(f"새 CSV 파일 생성 감지: {file_name}")
            name, _ = os.path.splitext(file_name)
            name = f'long_term_{int(name[-1]) - 1}.csv'
            # trace_generator를 호출하여 CSV 데이터를 기반으로 플롯 이미지를 생성하고, 이미지 파일 경로를 반환받음.
            image_file = trace_generator(name, save_dir, classifier_trace_dir)
            if image_file:
                #print(f"데이터 추출 및 이미지 생성 성공: {image_file}")
                # classifier_input_queue에 이미지 파일 경로 전달
                classifier_input_queue.put(image_file)
            else:
                pass
                #print(f"{file_name} 데이터 추출 실패.")

def run_csv_watcher(stop_event, watch_dir):
    event_handler = CSVCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=watch_dir, recursive=False)
    observer.start()
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    observer.stop()
    observer.join()

# --- LLM 피드백 처리 함수 ---
def run_feedback(ocr_queue, stop_event):
    """
    ocr_queue에서 label과 text를 포함한 아이템을 꺼내,
    feedback.py의 generate_feedback 함수를 호출하여 피드백 텍스트를 생성하고 출력합니다.
    """
    while not stop_event.is_set():
        try:
            item = ocr_queue.get(timeout=1)
            label = item.get("label", "unknown")
            text = item.get("text", "")
            start_time = time.time()
            # label에 따라 집중 상태를 결정 (예: "fixation" 또는 "regression")
            feedback_text = generate_feedback(text, label)
            print("Feedback:", feedback_text)
            ocr_queue.task_done()
            end_time = time.time()
            print(end_time - start_time)
        except queue.Empty:
            continue
        except Exception as e:
            print("Feedback error:", e)
            continue

def monitor_ocr_queue(ocr_queue, stop_event):
    while not stop_event.is_set():
        qsize = ocr_queue.qsize()
        print("Monitor: 현재 ocr_queue 크기 =", qsize)
        # 혹은 ocr_queue 내부 데이터를 peek하거나 get 후 다시 put하는 방식으로 내용을 확인할 수 있음
        time.sleep(5)

# --- Main ---
def main():
    jay_python = r"C:\Anaconda\envs\Jay\python.exe"  # 실제 경로에 맞게 수정

    # Calibration check
    if not os.listdir(cal_dir):
        print("Calibration directory is empty. Running calibration script...")
        calibration_cmd = [jay_python, os.path.join(data_generation, "one_point_calibration.py"), "--cal_dir", cal_dir]
        subprocess.run(calibration_cmd)

    print('end of calibration verification')
    realtime_tracking_path = os.path.join(data_generation, "realtime_gaze_tracking.py")
    proc = subprocess.Popen(
        [jay_python, realtime_tracking_path, "--data_dir", save_dir, "--cal_dir", cal_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print('started tracking your eyes')
    
    # 1. tracking data 읽기 스레드
    reader_thread = threading.Thread(target=read_tracking_data, args=(proc, tracking_queue, stop_event), daemon=True)
    reader_thread.start()
    
    # 2. stderr 읽기 스레드
    err_thread = threading.Thread(target=read_tracking_stderr, args=(proc, stop_event), daemon=True)
    err_thread.start()

    # 3. detection_processor 스레드
    detector_thread = threading.Thread(target=detection_processor, args=(tracking_queue, detection_queue, stop_event), daemon=True)
    detector_thread.start()
    print('started detecting the information')

    # 4. fixation_group_processor 스레드
    fixation_thread = threading.Thread(target=fixation_group_processor, args=(detection_queue, fixation_queue, stop_event, 1.0), daemon=True)
    fixation_thread.start()
    print('started fixation group processing')

    # 5. 플롯 저장 스레드
    #plot_thread = threading.Thread(target=save_output_plot, args=(regression_plot_queue, fixation_plot_queue, 5, os.path.join('Regression_plot', "output_plot.png"), stop_event), daemon=True)
    #plot_thread.start()

    # 6. Regression 로깅 스레드
    log_thread = threading.Thread(target=logging_regression, args=(stop_event,), daemon=True)
    log_thread.start()

    # 7. 스크린샷 캡처 스레드
    image_thread = threading.Thread(target=capture_screenshots_worker, args=(regression_queue, fixation_queue, screenshot_queue, stop_event), daemon=True)
    image_thread.start()

    # 8. 스크린샷 저장 스레드
    screenshot_thread = threading.Thread(target=save_screenshot, args=(screenshot_queue, os.path.join(screenshot_dir, 'regression'), os.path.join(screenshot_dir, 'fixation'), stop_event), daemon=True)
    screenshot_thread.start()

    # 9. classifier 스레드 (classifier_input_queue에서 이미지 파일 경로를 받아 처리)
    classification_thread = threading.Thread(target=run_classifier, args=(classifier_input_queue, classification_queue, stop_event), daemon=True)
    classification_thread.start()

    # 10. CSV 파일 생성 감지를 위한 watchdog 스레드
    csv_watchdog_thread = threading.Thread(target=run_csv_watcher, args=(stop_event, save_dir), daemon=True)
    csv_watchdog_thread.start()

    # 11. OCR 처리 스레드: screenshot_queue를 입력으로 받아 ocr_queue에 OCR 결과 전달

    ocr_thread = threading.Thread(target=process_screenshot_queue, args=(screenshot_queue, ocr_queue, stop_event, False), daemon=True)
    ocr_thread.start()

    # 12. LLM 피드백 처리 스레드: ocr_queue의 결과를 입력으로 받아 피드백 생성 후 텍스트 출력
    feedback_thread = threading.Thread(target=run_feedback, args=(ocr_queue, stop_event), daemon=True)
    feedback_thread.start()

    # 13. OCR 결과 모니터링 스레드
    #monitor_thread = threading.Thread(target=monitor_ocr_queue, args=(ocr_queue, stop_event), daemon=True)
    #monitor_thread.start()

    try:
        while not stop_event.is_set():
            try:
                result = fixation_queue.get(timeout=1)
                #print("fixation group:", result)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("Control_new.py 종료 요청됨.")
        stop_event.set()
    finally:
        stop_event.set()
        proc.terminate()
        reader_thread.join()
        detector_thread.join()
        err_thread.join()
        #plot_thread.join()
        log_thread.join()
        image_thread.join()
        screenshot_thread.join()
        classification_thread.join()
        csv_watchdog_thread.join()
        ocr_thread.join()
        feedback_thread.join()
        #monitor_thread.join()

if __name__ == '__main__':
    message = input('Sure that you want to run initialization?\n([Y]/N):')
    if message.upper() == 'Y' or message == '':
        from Init import initialization
        initialization(os.getcwd())
    else:
        print('Running the script without initialization')
    main()
