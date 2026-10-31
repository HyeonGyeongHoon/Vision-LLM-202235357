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


def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    weights_length = weights.size  # 가중치 배열 크기
    
    height = image.shape[0]  # 이미지 높이
    width = image.shape[1]  # 이미지 너비
    
    output = np.zeros((height,width), dtype=np.float32)  # 결과 배열 생성

    for output_y in range(height):  # 세로
        for output_x in range(width):  # 가로
            for shift in range(weights_length):  # 가중치 배열 크기 만큼 반복
                # TODO: x값 shift
                input_x = output_x - shift
                # 픽셀 범위 넘어가는 것 방지
                if input_x < 0 :
                    break
                # TODO: shift한 픽셀을 누적
                output[output_y,output_x] += image[output_y,input_x] * weights[shift]
                # TODO: 출력 배열에 누적 픽셀 저장

    return output


def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    weights_length = weights.size

    height = image.shape[0]  # 이미지 높이
    width = image.shape[1]  # 이미지 너비

    output = cp.zeros((height, width), dtype=cp.float32)  # 출력 배열 생성

    for shift in range(weights_length):
        # TODO: 배열 slicing
        input_region = image[:,:width-shift]  # 입력 영역
        output_region = output[:,shift:]  # 출력 영역

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region = input_region * weights[shift]  # 출력 영역 계산
        output[:,shift:] += output_region  # 출력 배열에 누적 연산

    return output


def main():
    # 0. 이미지 불러오기
    input_np = load_img_np()


    # 1. 가중치 설정
    weights_np = np.linspace(
        1.0,
        0.05,
        TRAIL_LENGTH,
        dtype=np.float32
    )  # 이동 거리가 멀어질수록 가중치를 작게 설정
    weights_np /= np.sum(weights_np)  # 전체 가중치의 합을 1로 정규화


    # 2. Numpy CPU 처리
    cpu_start = time.perf_counter()

    output_np = motion_trail_np(
        image=input_np,
        weights=weights_np
    )

    cpu_end = time.perf_counter()

    cpu_time = cpu_end - cpu_start


    # 3. CuPy 배열 준비
    upload_start = time.perf_counter()

    input_cp = cp.asarray(input_np)  # asarray() 사용
    weights_cp = cp.asarray(weights_np)  # asarray() 사용

    cp.cuda.Stream.null.synchronize()

    upload_end = time.perf_counter()

    upload_time = upload_end - upload_start


    # 4. GPU 워밍업
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np[:4])

    _ = motion_trail_cp(
        image=warmup_image,
        weights=warmup_weights
    )

    cp.cuda.Stream.null.synchronize()


    # 5. CuPy GPU 처리
    cp.cuda.Stream.null.synchronize()

    gpu_start = time.perf_counter()

    output_cp = motion_trail_cp(
        image=input_cp,
        weights=weights_cp
    )

    cp.cuda.Stream.null.synchronize()

    gpu_end = time.perf_counter()

    gpu_time = gpu_end - gpu_start


    # 6. GPU 결과를 CPU로 가져오기
    download_start = time.perf_counter()

    output_cp_np = cp.asnumpy(output_cp)  # cp.asnumpy 사용

    download_end = time.perf_counter()

    download_time = download_end - download_start


    # 7. 결과 비교
    compute_speedup = cpu_time / gpu_time

    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time


    # 8. 결과 출력
    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")


    # 9. 이미지 출력
    # Original Image, Motion Trail(CPU), Motion Trail(GPU) 비교
    plt.figure(figsize=(18,12))

    plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
    plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.suptitle("Original Image, Motion Trail(CPU), Motion Trail(GPU) 비교")

    # Motion Trail(GPU) 크게 보기
    plt.figure(figsize=(12,8))
    plt.imshow(output_cp_np, cmap="gray")
    plt.axis("off")

    plt.suptitle("Motion Trail(GPU)")

    # 모든 창 출력
    plt.show()

if __name__ == "__main__":
    main()