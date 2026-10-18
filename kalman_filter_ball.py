from numpy.linalg import inv
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

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

# 객체 최초 검출 여부 확인용 boolean
found = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2에서 프레임의 색공간은 BGR이기 때문에 LAB으로 변환하기 위해 RGB로 변환
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)  # LAB으로 변환

    l, a, b = cv2.split(frame_lab)  # 채널 분리
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
    l_clahe_enhanced = clahe.apply(l)  # L 채널에 CLAHE 적용 후 강화된 이미지 생성
    frame_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    mask_enhanced = cv2.inRange(frame_enhanced_lab, green_lower, green_upper)  # 강화된 이미지로 binary mask 생성

    frame_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)
    frame_mask_closed = cv2.morphologyEx(frame_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    final_frame_mask = frame_mask_closed

    contour_lst, _ = cv2.findContours(
    final_frame_mask,          # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    if len(contour_lst) > 0:
        # 가장 큰 contour
        largest_contour = max(contour_lst, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_frame_mask)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)


        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        center = (cX, cY)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255,0,0), 3)
        cv2.circle(frame, center, 5, (255,0,0), -1)

        # 객체 최초 검출
        if not found:
            found = True

    # x_bel, y_bel = 0., 0.
    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    if found: 
        # 예측한 객체 위치에 노란 원 overlay
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0,255,255), 3)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0,255,255), -1)

    cv2.imshow("Green ball tracing", frame)

cap.release()
cv2.destroyAllWindows()