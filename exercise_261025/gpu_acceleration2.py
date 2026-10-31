# 라이브러리 호출
import time
import cv2
import numpy as np 
import cupy as cp
import matplotlib.pyplot as plt

# 글로벌 변수 정의
IMAGE_PATH = "../src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

# 이미지 불러오기
def load_img_np():
    img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(
        img_gray,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    input_np = img_gray.astype(np.float32) / 255.0

    return input_np

# relu 함수 정의
def relu(x):
    return np.maximum(0, x)

# Numpy를 사용하여 2차원 convolution 함수 정의
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

# CuPy를 사용하여 2차원 convolution 함수 정의
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : cp.ndarray
        입력 이미지, shape = (height, width)
    kernel : cp.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : cp.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    # 커널 요소별로 연산
    mid_idx = kernel_width // 2  # kernel_width == kernel_height
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # kernel의 인덱스 값을 토대로 연산할 padded_image 크기 슬라이싱
            start_y = kernel_y
            end_y = start_y + output_height
            start_x = kernel_x
            end_x = start_x + output_width
            output += padded_image[start_y:end_y, start_x: end_x] * kernel[kernel_y, kernel_x]

    # 편향 적용
    output = output + bias
    return output

def main():
    # 0. 이미지 불러오기
    input_np = load_img_np()

    # 1. NumPy CPU 처리 : 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
    # x축 방향 edge (세로 선) 검출 필터
    dx_edge_filter = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        dtype=np.float32
    )

    # y축 방향 edge (가로 선) 검출 필터
    dy_edge_filter = np.array(
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        dtype=np.float32
    )

    # 편향 설정 : 너무 약한 특징 (0.2 이하)은 무시
    dx_bias = -0.2
    dy_bias = -0.2

    cpu_start = time.perf_counter()

    output_dx_np = convolution2d(
        image=input_np,
        kernel=dx_edge_filter,
        bias=dx_bias,
        stride=1,
        padding=1
    )
    output_dy_np = convolution2d(
        image=input_np,
        kernel=dy_edge_filter,
        bias=dy_bias,
        stride=1,
        padding=1
    )

    cpu_end = time.perf_counter()

    cpu_time = cpu_end - cpu_start

    relu_dx_np = relu(output_dx_np)
    relu_dy_np = relu(output_dy_np)


    # 2. CuPy GPU 처리 : convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
    input_cp = cp.asarray(input_np)  # asarray() 사용

    dx_edge_filter_cp = cp.asarray(dx_edge_filter)
    dy_edge_filter_cp = cp.asarray(dy_edge_filter)

    # 워밍업
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_filter = cp.zeros((3, 3), dtype=cp.float32)
    _ = convolution2d_cp(
        image=warmup_image,
        kernel=warmup_filter
    )

    # gpu 시간 측정 시작
    cp.cuda.Stream.null.synchronize()
    gpu_start = time.perf_counter()

    output_dx_cp = convolution2d_cp(
        image=input_cp,
        kernel=dx_edge_filter_cp,
        bias=dx_bias,
        stride=1,
        padding=0
    )
    output_dy_cp = convolution2d_cp(
        image=input_cp,
        kernel=dy_edge_filter_cp,
        bias=dy_bias,
        stride=1,
        padding=0
    )

    # gpu 시간 측정 종료
    cp.cuda.Stream.null.synchronize()
    gpu_end = time.perf_counter()

    gpu_time = gpu_end - gpu_start

    output_dx_cp_np = cp.asnumpy(output_dx_cp)  # cp.asnumpy 사용
    output_dy_cp_np = cp.asnumpy(output_dy_cp)  # cp.asnumpy 사용

    relu_dx_cp_np = relu(output_dx_cp_np)
    relu_dy_cp_np = relu(output_dy_cp_np)

    # 3. 결과 출력 : CPU/GPU 결과 비교 및 출력
    compute_speedup = cpu_time / gpu_time

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")

    # 4. 이미지 출력
    plt.figure(figsize=(18,8))

    plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()

if __name__ == "__main__":
    main()