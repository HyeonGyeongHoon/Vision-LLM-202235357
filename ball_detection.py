# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
# TODO: 각 프레임 이미지를 전처리
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay

import cv2
import numpy as np

# pipeline 정의
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

green_lower = np.array([30, 60, 90], dtype=np.uint8)  # 초록색 하한값
green_upper = np.array([230, 115, 180], dtype=np.uint8)  # 초록색 상한값

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 둥근 모양을 위한 커널 생성

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)  # 카메라 객체 불러오기

def get_enhanced_frame(frame):
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)  # LAB으로 변환

    l, a, b = cv2.split(frame_lab)  # 채널 분리
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
    l_clahe_enhanced = clahe.apply(l)  # L 채널에 CLAHE 적용 후 강화된 이미지 생성
    frame_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    return frame_enhanced_lab

def get_frame_mask(mask):

    frame_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)
    frame_mask_closed = cv2.morphologyEx(frame_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return frame_mask_closed

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2에서 프레임의 색공간은 BGR이기 때문에 LAB으로 변환하기 위해 RGB로 변환

    frame_enhanced_lab = get_enhanced_frame(frame_rgb)
    mask_enhanced = cv2.inRange(frame_enhanced_lab, green_lower, green_upper)  # 강화된 이미지로 binary mask 생성

    final_frame_mask = get_frame_mask(mask_enhanced)

    contours, _ = cv2.findContours(
    final_frame_mask,          # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )
    
    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_frame_mask)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)

        # 노란색 무게중심 기반 원
        cv2.circle(frame, center_circle, radius, (0,255,255), 3)
        cv2.circle(frame, center_circle, 5, (0,255,255), -1)

    cv2.imshow("Green ball Detection", frame)

cap.release()
cv2.destroyAllWindows()