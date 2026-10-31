import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.data

from random import randint, gauss

def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))

def create_random_bbox():
    hubble = skimage.data.hubble_deep_field()

    N = 20

    boxes = []
    scores = []

    for i in range(N):
        boxes.append(random_bbox(hubble))
        scores.append(random_conf())

    return boxes, scores

# BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x+w, y+h
    return (x1, y1, x2, y2)

# IoU 계산 함수 정의
def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    intersection_area = w * h
    union_area = (xa_2 - xa_1) * (ya_2 - ya_1) + (xb_2 - xb_1) * (yb_2 - yb_1) - intersection_area

    if union_area == 0:
        return 0

    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area


# 신뢰도가 너무 낮은 BBox 제거 (Confidence Thresholding) 함수
def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)
    
    return pruned_boxes, pruned_scores


# NMS 함수 정의
def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    nms_boxes = []
    nms_scores = []

    # np.argsort() 사용해서 오름차순 정렬함
    sort_scores = list(np.argsort(scores)) # pop()함수 쓸려고 다시 리스트로

    # 2. 모든 후보 박스를 검사할 때까지 반복
    while len(sort_scores) > 0:
        max_idx = sort_scores.pop()  # 마지막 요소가 가장 큰 값의 인덱스

        max_box = boxes[max_idx]
        max_score = scores[max_idx]

        nms_boxes.append(max_box)
        nms_scores.append(max_score)

        remaining_indexs = []  # 신뢰도가 가장 높은 박스와 iou 계산하고 임계값보다 낮은 박스의 인데스를 저장하는 리스트
        for idx in sort_scores:
            iou = calculate_iou(max_box, boxes[idx])
            if iou < iou_threshold:
                remaining_indexs.append(idx)

        sort_scores = remaining_indexs

    return nms_boxes, nms_scores


def main():
    # 1. 이미지 불러오기
    hubble = skimage.data.hubble_deep_field()  # 원본 이미지
    hubble_nms = skimage.data.hubble_deep_field()  # NMS 적용할 이미지

    # 2. 랜덤 bbox 생성
    boxes, scores = create_random_bbox()

    # 3. 낮은 confidence 제거 및 NMS 구현
    new_boxes, new_scores = prune_low_conf(boxes, scores)
    new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

    # 4. BBox 그리기
    # 원본 BBox
    for i, box in enumerate(boxes):
        draw_bbox(
            hubble,
            (box[0], box[1]),
            (box[2], box[3]),
            (randint(0,255), randint(0,255), randint(0,255)),
            txt=f"{scores[i]:.2f}"
        )
    # NMS 적용한 BBox
    for i, box in enumerate(new_boxes):
        draw_bbox(
            hubble_nms,
            (box[0], box[1]),
            (box[2], box[3]),
            (randint(0,255), randint(0,255), randint(0,255)),
            txt=f"{scores[i]:.2f}"
        )

    # 5. 이미지 출력
    plt.figure(figsize=(16,10))
    plt.subplot(1,2,1), plt.imshow(hubble_nms)
    plt.title("Original BBox")
    plt.subplot(1,2,2), plt.imshow(hubble)
    plt.title("Custom NMS Result")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.show()