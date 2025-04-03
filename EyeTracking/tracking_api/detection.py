import queue

def detection_processor(tracking_queue, detection_queue, fixation_threshold=50, regression_threshold=100, y_tolerance=30):
    """
    tracking_queue에서 데이터를 꺼내어, 이전 이벤트와 비교해 간단한 판정을 내고 결과를 detection_queue에 저장합니다.
    
    - 같은 줄인지 판단하기 위해 y 좌표 차이가 y_tolerance 이하인 경우에만 비교합니다.
    - x 좌표가 fixation_threshold 이하의 변화이면 Fixation,
      regression_threshold 이상의 감소이면 Regression,
      그 외에는 Saccade로 판정합니다.
    - 이전 이벤트가 없으면 기본 라벨을 부여합니다.
    """
    prev_event = None
    while True:
        try:
            event = tracking_queue.get(timeout=1)
        except queue.Empty:
            continue

        # 이전 이벤트가 없으면 기본 라벨 지정
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
        prev_event = event
