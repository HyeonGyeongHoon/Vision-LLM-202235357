import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_rgb_image():
    """이미지 파일을 읽어와 RGB 배열로 변환"""
    basketball = cv2.imread('../src/images/basketball_crop.jpg')
    basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)
    return basketball_rgb


def do_clahe(ball_image):
    """LAB색공간에서 L 채널만 따로 추출한 뒤 CLAHE를 적용하고 색상 범위 지정 후 마스크 생성 및 반환"""
    lower_basketball_lab = np.array([70, 130, 70])
    upper_basketball_lab = np.array([180, 230, 185])

    basketball_lab = cv2.cvtColor(ball_image, cv2.COLOR_RGB2LAB)  # LAB 색공간으로 변환
    l, a, b = cv2.split(basketball_lab)  # 각 채널 분리

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
    l_clahe_enhanced = clahe.apply(l)  # L 채널에 CLAHE 적용 후 강화된 이미지 생성
    basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)  # 강화된 이미지로 binary mask 생성

    return mask_enhanced


def mophological_operation(mask):
    """모폴로지 열림 연산(3회), 닫힘 연산(1회)를 적용하여 마스크의 노이즈 제거"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ball_mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    ball_mask_closed = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel)

    return ball_mask_closed


def get_contours(final_ball_mask):
    """노이즈가 제거된 이진 마스크에서 외곽선 좌표를 추출"""
    # 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
    contours, _ = cv2.findContours(
        final_ball_mask,          # binary mask
        cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    return contours


def draw_contours(rgb_image, final_mask, contours):
    """추출된 모든 외곽선을 이미지 위에 그리기"""
    # Contour를 그릴 원본 이미지 copy
    final_img = rgb_image.copy()

    if len(contours) > 0:
        # Binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(final_img)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

        # Contour 그리기 (초록색 외곽선)
        cv2.drawContours(final_img, contours, -1, (0, 255, 0), 2)

    plt.imshow(final_img)
    plt.title("Draw Contours")
    plt.axis("off")
    plt.show()


def draw_ball_contours(rgb_image, final_mask, contours):
    """가장 큰 외곽선을 찾아내어 농구공 영역을 시각화하고 그 영역의 외곽선을 반환"""
    # Contour를 그릴 원본 이미지 copy
    final_img = rgb_image.copy()

    ball_contour = None  # 공의 외곽선 미검출 시 안전하게 반환
    if len(contours) > 0:
        # 가장 큰 contour
        ball_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        ball_mask = np.zeros_like(final_mask)
        cv2.drawContours(
            ball_mask,
            [ball_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(final_img)
        mask_color[:, :, 1] = ball_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

        # Contour 그리기 (초록색 외곽선)
        cv2.drawContours(final_img, [ball_contour], -1, (0, 255, 0), 2)

    plt.imshow(final_img)
    plt.title("Final Basketball Section & Contour")
    plt.axis("off")
    plt.show()

    return ball_contour


ball_image = load_rgb_image()
enhanced_mask = do_clahe(ball_image)
final_mask = mophological_operation(enhanced_mask)  # Morphological operation을 적용하여 최종 binary mask 생성
ball_image_contours = get_contours(final_mask)
draw_contours(ball_image, final_mask, ball_image_contours)  # Binary mask로 외곽선 검출
draw_ball_contours(ball_image, final_mask, ball_image_contours)  # 최종 농구공 영역 및 외곽선 선택