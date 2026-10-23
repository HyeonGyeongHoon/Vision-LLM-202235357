import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 파란색 칫솔 검출용 LAB 임계값
LOWER_COLOR_BOUND = np.array([20, 120, 20], dtype=np.uint8)
UPPER_COLOR_BOUND = np.array([200, 170, 110], dtype=np.uint8)

MIN_CONTOUR_AREA = 100  # 칫솔만 검출하기 위한 최소 외곽선 면적 (픽셀 단위)

# Kalman Filter 상태/측정 차원
STATE_DIM = 4  # [x, y, vx, vy]^T
MEASURE_DIM = 2  # [x, y]^T

TARGET_BRUSH_TIME = 180.0  # 목표 양치 시간 (180초 = 3분)

MIN_BRUSH_SPEED = 2.0  # 유효 양치 동작 최소 속력 (픽셀/프레임)


# 1. 파란색 칫솔 영역 검출
def do_clahe(frame_bgr):
    """LAB 색공간 변환 및 L채널 CLAHE 적용 후 이진 마스크 생성 및 반환"""
    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    channel_l, a, b = cv2.split(frame_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(channel_l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))

    mask = cv2.inRange(enhanced_lab, LOWER_COLOR_BOUND, UPPER_COLOR_BOUND)
    return mask

def mophological_operation(mask):
    """모폴로지 열림(1회) 및 닫힘(2회) 연산으로 잡음 정제 및 최종 마스크 반환"""
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return mask_closed

def get_contours(final_mask):
    """최종 마스크에서 외곽선 추출 후 반환"""
    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def get_toothbrush_centroid(contours):
    """최대 면적 컨투어 선별 후 무게중심 좌표 (x, y)랑 가장 큰 외곽선 반환"""
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:  # 가장 큰 외곽선의 면적이 최소 면적보다 큰지 비교
        return None, None

    # Moments를 통해 무게중심 (Centroid) 산출
    m = cv2.moments(largest_contour)  # 큰 외곽선의 모멘트 구하기
    if m["m00"] == 0:
        return None, None

    cx = int(m["m10"] / m["m00"])  # 무게중심의 x좌표
    cy = int(m["m01"] / m["m00"])  # 무게중심의 y좌표
    return (cx, cy), largest_contour


# 2. Kalman Filter 설정
def init_kalman_filter():
    """등속도 운동 모델 기반 2D 칼만 필터 객체 초기화 하는 함수. 칼만 필터 반환"""
    kalman = cv2.KalmanFilter(STATE_DIM, MEASURE_DIM)

    # Transition Matrix (상태 전이 행렬)
    kalman.transitionMatrix = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Measurement Matrix (측정 행렬)
    # 측정값으로부터 x, y만 관측
    kalman.measurementMatrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    # Process Noise Covariance Matrix (프로세스 노이즈 공분산)
    # 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
    kalman.processNoiseCov = np.array(
        [
            [1e-2, 0, 0, 0],
            [0, 1e-2, 0, 0],
            [0, 0, 5e-2, 0],
            [0, 0, 0, 5e-2],
        ],
        dtype=np.float32,
    )

    # Measurement Noise Covariance Matrix (측정 노이즈 공분산)
    # 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
    kalman.measurementNoiseCov = np.eye(MEASURE_DIM, dtype=np.float32) * 1.0

    # Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
    kalman.errorCovPost = np.eye(STATE_DIM, dtype=np.float32)

    return kalman

# 3. MediaPipe Task API 얼굴 및 입 구역 검출 모듈
def init_mediapipe_face_detector(model_path):
    """MediaPipe Tasks API FaceDetector 객체를 생성하여 반환"""
    base_option = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_option,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)

def detect_face_and_mouth_zone(frame_bgr, detector):
    """얼굴 및 하단 50%의 입 주변 네모 검출 """
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # 프레임 얼굴 탐지
    detection_result = detector.detect(mp_image)

    # 얼굴이 한 명도 검출되지 않은 경우 안전하게 None 반환
    if not detection_result.detections:
        return None, None

    # 가장 높은 신뢰도를 가진 첫 번째 얼굴의 바운딩 박스 추출
    bbox = detection_result.detections[0].bounding_box

    # 바운딩 박스 좌표 추출 및 화면 넘어가는 거 클립핑
    fx = max(0, bbox.origin_x)
    fy = max(0, bbox.origin_y)
    fw = min(bbox.width, frame_w - fx)
    fh = min(bbox.height, frame_h - fy)

    face_box = (fx, fy, fw, fh)  # 얼굴 영역
    mouth_zone = (fx, fy + fh // 2, fw, fh // 2)  # 입 주변 영역 : 얼굴 영역 하단 50%
    return face_box, mouth_zone

# 칫솔 무게 중심 좌표가 입 주변 영역에 들어오는 지 알려주는 함수
def is_point_inside_rect(point, rect):
    """좌표 (px, py)가 사각형 (rx, ry, rw, rh) 내부에 위치하는지 판정 후 참/거짓 반환"""
    px, py = point
    rx, ry, rw, rh = rect
    return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

# 메인 함수 시작
def main():
    # pipeline 정의
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(0)
    # 얼굴 인식 모델 불러오기
    detector = init_mediapipe_face_detector(model_path="../src/models/MediaPipe/detector.tflite")
    # Kalman Filter 초기화
    kalman = init_kalman_filter()

    # 타이머 관리 변수
    remaining_time = TARGET_BRUSH_TIME
    last_tick_time = time.time()

    # 객체 최초 검출 여부 확인용 boolean
    found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 거울 모드 좌우 반전
        frame = cv2.flip(frame, 1)

        # 
        current_time = time.time()
        delta_time = current_time - last_tick_time
        last_tick_time = current_time

        # 1. 신체 인식 (얼굴 및 입 주변 영역 검출)
        face_box, mouth_zone = detect_face_and_mouth_zone(frame, detector)

        # 2. 칫솔 색상 검출 및 무게중심 계산
        raw_mask = do_clahe(frame)
        clean_mask = mophological_operation(raw_mask)
        contours = get_contours(clean_mask)
        centroid, largest_contour = get_toothbrush_centroid(contours)    

        # 3. Kalman Filter 예측
        predicted_state = kalman.predict()  # 예측(prediction)
        pred_x = int(predicted_state[0, 0])
        pred_y = int(predicted_state[1, 0])

        # 4. Kalman Filter 보정
        if centroid is not None:
            cx, cy = centroid
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not found:
                # 첫 번째 관측값으로 상태 벡터 초기화 (튐 방지)
                kalman.statePost = np.array(
                    [[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]],
                    dtype=np.float32
                )
                found = True

            estimated = kalman.correct(measurement)
            tracked_x = int(estimated[0, 0])
            tracked_y = int(estimated[1, 0])
            tracker_status = "LOCKED"
        else:
            # 가려짐 상황(Occlusion): 관측 보정 없이 예측 위치 채택
            tracked_x = pred_x
            tracked_y = pred_y
            tracker_status = "PREDICTING"

        # 5. 상태 벡터로부터 속도 및 물리량 도출
        vx = kalman.statePost[2, 0]
        vy = kalman.statePost[3, 0]
        speed = np.sqrt(vx * vx + vy * vy)
        tracked_point = (tracked_x, tracked_y)

        # 6. 양치질 상태 판정 및 타이머 갱신
        brushing_active = False
        if found and (mouth_zone is not None):
            is_inside = is_point_inside_rect(tracked_point, mouth_zone)
            is_moving = speed >= MIN_BRUSH_SPEED

            # 입 구역 내부에서 사전에 정의한 기준 속도 이상으로 움직일 때만 타이머 카운트다운 시작
            if is_inside and is_moving:
                brushing_active = True
                remaining_time = max(0.0, remaining_time - delta_time)

        # 7. 시각화
        # 1) 얼굴 영역 사각형
        if face_box is not None:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # 2) 입 주변 영역 사각형 (진입 시 초록색, 대기 시 노란색)
        if mouth_zone is not None:
            mx, my, mw, mh = mouth_zone
            zone_color = (0, 255, 0) if brushing_active else (0, 255, 255)
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), zone_color, 2)

        # 3) 칫솔 외곽선 (카메라에 직접 검출될 때만 표시)
        if centroid is not None:
            cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 1)

        # 4) 칼만 필터 현재 추적점 (LOCKED: 파란색, PREDICTING: 주황색)
        if found:
            point_color = (255, 0, 0) if tracker_status == "LOCKED" else (0, 165, 255)
            cv2.circle(frame, tracked_point, 6, point_color, -1)

        # 5) 타이머 표시
        minutes = int(remaining_time) // 60  # 분
        seconds = int(remaining_time) % 60  # 초
        timer_text = f"Timer: {minutes:02d}:{seconds:02d}"  # 숫자 앞에 0 붙이기

        face_text = "Face: DETECTED" if face_box is not None else "Face: NONE"
        status_text = "Action: BRUSHING (ACTIVE)" if brushing_active else "Action: PAUSED"
        status_color = (0, 255, 0) if brushing_active else (0, 0, 255)

        cv2.putText(
            frame,
            face_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if face_box is not None else (0, 0, 255),
            2
        )
        cv2.putText(
            frame,
            f"Tracker: {tracker_status if found else 'WAITING'}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            point_color if found else (150, 150, 150),
            2
        )
        cv2.putText(
            frame,
            f"Speed: {speed:.1f} px/f",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
        cv2.putText(
            frame,
            status_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )
        cv2.putText(
            frame,
            timer_text,
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Toothbrush Smart Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()