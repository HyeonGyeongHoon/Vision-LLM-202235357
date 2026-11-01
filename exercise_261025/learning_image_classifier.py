import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 난수 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 글로벌 상수 정의
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)
CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

EPOCHS = 5  # 모델 학습 수

# 데이터 전처리 함수
def preprocessing_data():
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    return train_transform, test_transform

# CIFAR-10 데이터셋 설치 및 불러오기
def load_cifar_10_datasets(train_transform, test_transform ):
    train_dataset = datasets.CIFAR10(
        root="src/datasets",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="src/datasets",
        train=False,
        transform=test_transform,
        download=True,
    )

    return train_dataset, test_dataset

# DataLoader 생성
def create_dataloder(train_dataset, test_dataset):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader

# CNN 모델 정의
# 1) CNN Block 정의
class ConvBlock(nn.Module):
    """
    Convolution을 이용해 특징을 추출하고 Max Pooling을 적용하는 재사용 가능한 CNN 블록.

    기본 구조:
    Conv2d → BatchNorm2d → ReLU
    → Conv2d → BatchNorm2d → ReLU
    → MaxPool2d
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

         # Block을 구성할 Layer들을 순서대로 저장
        layers = [

            # 첫 번째 Convolution (특징 추출)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 각 출력 채널의 값 분포를 조정하여 학습 안정화
            nn.BatchNorm2d(out_channels),

            # ReLU 활성화 함수
            nn.ReLU(),

            # 두 번째 Convolution (특징 추가적으로 추출)
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 두 번째 Convolution의 출력을 정규화
            nn.BatchNorm2d(out_channels),

            # 두 번째 활성화 함수를 적용
            nn.ReLU(),

            # Max Pooling 적용하여 다운스케일
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
            
        ]

        # 리스트에 저장한 Layer들을 순서대로 실행하는 하나의 Sequential Module로 구성
        self.block = nn.Sequential(*layers)

    # 입력 x를 self.block의 Layer에 순서대로 전달
    def forward(self, x):
        return self.block(x)

# CIFAR-10 분류 모델 정의
class CIFAR10CNN(nn.Module):
    """
    CIFAR-10의 32x32 RGB 이미지를 10개 클래스로 분류하는 CNN 모델.

    전체 구조:
    ConvBlock 1
    → ConvBlock 2
    → ConvBlock 3
    → Global Average Pooling
    → Dropout
    → Linear
    """

    def __init__(self):
        super().__init__()

        # 이미지에서 특징을 추출하는 부분 (ConvBlock 1 → ConvBlock 2 → ConvBlock 3)
        self.features = nn.Sequential(

            
            ConvBlock(                 # 출력: [Batch, 64, 16, 16]
                in_channels=3,
                out_channels=64,
            ),

            
            ConvBlock(                 # 출력: [Batch, 128, 8, 8]
                in_channels=64,
                out_channels=128,
            ),

            
            ConvBlock(                 # 출력: [Batch, 256, 4, 4]
                in_channels=128,
                out_channels=256,
            ),

        )

        # 각 4×4 Feature Map의 평균을 계산하여 공간 크기를 1×1로 압축
        self.global_average_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)     # 출력: [Batch, 256, 1, 1]
                )
        )

        # 추출된 특징을 이용해 최종 클래스를 예측
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),         # 출력: 256

            nn.Linear(                 # 출력: 10
                in_features=256,
                out_features=10,
            ),
        )

    # 입력 이미지: [Batch, 3, 32, 32]
    def forward(self, x):
        # 세 개의 ConvBlock을 통과하며 특징을 추출
        # 출력: [Batch, 256, 4, 4]
        x = self.features(x)

        # 각 채널의 4×4 값을 평균내어 1×1로 압축
        # 출력: [Batch, 256, 1, 1]
        x = self.global_average_pool(x)

        # Tensor를 2차원으로 펼침
        # [Batch, 256, 1, 1] → [Batch, 256]
        x = torch.flatten(x, start_dim=1)

        # 각 클래스에 대한 점수인 Logit을 계산
        # 출력: [Batch, 10]
        logits = self.classifier(x)

        return logits

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # 장치(CPU/GPU)로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # 장치(CPU/GPU)로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


# GPU Warm-up
def gpu_warm_up(device, model):
    warmup_input = torch.zeros(64, 3, 32, 32, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)

    torch.cuda.synchronize()

def learning_model(model, train_loader, criterion, optimizer, device, test_loader):
    # CNN 모델 학습

    # 학습 기록 저장 공간 준비
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # 전체 학습 시간 측정 시작
    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    # TODO: 5번의 Epoch 동안 학습 및 평가 진행
    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )
        
    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time

    # TODO: 전체 학습 시간 계산 및 출력
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history

# 모델 저장
def save_model(model, optimizer, test_accuracy_history) :

    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    torch.save(
        save_data,
        "../src/models/CIFAR10/CIFAR10_CNN.pth",
    )

def main():
    # 1. 데이터 전처리
    train_transform, test_transform = preprocessing_data()

    # 2. 데이터셋 불러오기
    train_dataset, test_dataset = load_cifar_10_datasets(train_transform, test_transform)

    # 3. 데이터로더 생성
    train_loader, test_loader = create_dataloder(train_dataset, test_dataset)

    # 4. 모델 생성 및 GPU 지정
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다."
        )

    device = torch.device("cuda")
    model = CIFAR10CNN().to(device)

    # 5. 손실 함수 및 Optimizer 설정
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    # 6. GPU 웜업
    gpu_warm_up(device, model)

    # 7. CNN 모델 학습
    _, _, _, test_accuracy_history = learning_model(model, train_loader, criterion, optimizer, device, test_loader)

    # 8. 모델 저장
    save_model(model, optimizer, test_accuracy_history)

if __name__ == "__main__":
    main()