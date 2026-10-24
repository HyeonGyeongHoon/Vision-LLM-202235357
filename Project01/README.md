# Project01 제출

## 프로젝트 설명

### 1. 개요
- 온디바이스(Jetson) 환경에서 CSI 카메라 영상만을 활용한 스마트 칫솔질 인식 및 3분 양치 타이머 모니터링 시스템
- 별도 하드웨어 센서 없이 영상 처리(OpenCV)와 경량 AI(MediaPipe) 결합으로 양치 유효 동작 인식

### 2. 주요 기능 및 파이프라인
1. **CSI 카메라 영상 취득**
   - GStreamer 파이프라인(`nvarguscamerasrc`, `nvvidconv`)을 적용하여 Jetson 보드에서 1280x720 @ 30fps로 저지연 프레임 캡처
2. **신체 인식 및 Mouth Zone 설정**
   - MediaPipe Tasks API(`FaceDetector`)로 얼굴 영역 검출
   - 검출된 얼굴 바운딩 박스의 하단 50% 영역을 슬라이싱하여 실제 양치 판정 영역(Mouth Zone) 정의
3. **타깃(파란색 칫솔) 영역 검출 및 전처리**
   - 피부색 및 입술 간섭을 배제하기 위해 파란색 타깃 색상 추출 (LAB 색공간 임계값 적용)
   - L 채널 CLAHE 적용으로 화장실/실내 조명 반사 완화
   - 모폴로지 연산(타원형 커널 기반 열림 1회로 배경 노이즈 제거, 닫힘 2회로 물체 내부 빈틈 메우기) 수행
   - 외곽선(Contour) 모멘트(`cv2.moments`)를 계산하여 칫솔의 중심 좌표(Centroid) 도출
4. **Kalman Filter 기반 객체 추적**
   - 4차원 상태 벡터 `[x, y, vx, vy]^T` 기반의 등속도 운동 모델 구축
   - 센서 관측 노이즈 스무딩 및 칫솔이 입속/손에 가려지는 상황(Occlusion)에서 예측(`predict`) 좌표 유지
5. **양치 동작 판정 및 Delta-Time 타이머**
   - 판정 조건: 추적 좌표가 Mouth Zone 내부에 위치하며, 산출된 이동 속력(`speed`)이 기준값(2.0 px/frame) 이상일 때만 유효 양치 동작으로 인식
   - 시스템 연산 부하에 따른 FPS 변동 시에도 실제 시간이 일정하게 줄어들도록 `time.time()` 기반의 Delta-Time(경과 시간) 차감 방식 적용 (목표 시간: 180초)

<br>

## 실행 파일
- main.py: 프로젝트 실행 파일

<br>

## Visual Studio Code 가상 환경 생성 및 실행 절차
0. vscode 좌측 상단 File -> Open Folder에서 원하는 폴더 열기
1. Visual Studio Code에서 Ctrl + Shift + ~ 눌러 터미널 열기
2. python3-venv 패키지 설치:
```shell
sudo apt install python3-venv
```
3. 가상 환경 생성:
```shell
python3 -m venv --system-site-packages .venv
```
4. 가상 환경 활성화:
```shell
source .venv/bin/activate
```

5. 실습 파일 및 과제 파일 열기
```shell
code main.py
```
- 또는 New File.. 눌러서 파일 생성
6. 실습파일 편집
- VS Code 에디터 창에 코드 붙여넣기 후 Ctrl + S로 저장
7. 파이썬 파일 실행
- 에디터 우측 상단의 재생 버튼 클릭

<br>

## 개발 및 실행 환경 (Dependencies)
- 하드웨어: NVIDIA Jetson (JetPack 환경, YUYV 지원 USB 카메라 모듈)
- 설치 라이브러리:
  - `mediapipe` (MediaPipe Tasks Python API)
```bash
pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice
```
- **사전 준비 모델 파일**:
  - `detector.tflite` (MediaPipe BlazeFace Short Range 모델)
  - 저장 경로: `src/models/MediaPipe/detector.tflite`
```bash
wget -O src/models/MediaPipe/detector.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

<br>

## 기타 참조 사항
### 1. 주요 하이퍼파라미터 설정 근거
#### 1) 전처리 및 색상 검출
- 파란색 LAB 색상 범위 **(L: 20~200, a: 120~170, b: 20~110)**
    - 선정 이유 : 실제 칫솔 대신 파란색 연필로 테스트 진행
    - 채널 범위 지정 근거
        - L (명도) : 20 미만 그림자 영역과 200 이상 조명 반사 영역을 제외한 범위
        - a (초록-빨강) : 피부 및 입술의 붉은색 영역을 제외하기 위해 무채색인 중립값 부근 영역 범위
        - b (노랑-파랑) : 순수 파란 색의 계열만 분리하기 위해 파란색 영역 범위
    - 색상 선택기 : https://rakko.tools/ko/tools/color-picker
- CLAHE 파라미터 **(clipLimit=3.0, tileGridSize=(8, 8))**
    - 국소 타일(8x8) 단위로 명암 대비를 개선하여 실내 또는 화장실 조명 불균일 문제를 완화
    - clipLimit=3.0으로 상한선을 두어 배경 센서 노이즈가 과도하게 증폭되는 현상을 차단
#### 2) 모폴로지 연산
- 모폴로지 하이퍼 파라미터를 동적으로 테스트해보기 위해 트랙바를 사용하여 최적의 하이퍼 파라미터를 선택
- 타원형 구조화 요소 **(cv2.MORPH_ELLIPSE, 크기 (5, 5))** : 모서리 깨짐 없이 영역이 선명하게 나옴
- 열림 연산 **(cv2.MORPH_OPEN, iterations=1)** : 1회 초과 시 연필 영역이 끊어지는 결과가 나옴
- 닫힘 연산 **(cv2.MORPH_CLOSE, iterations=2)** : 2회에서 가장 영역이 끊어지지 않고 선명하게 나옴
- 최소 외곽선 유효 면적 **(MIN_CONTOUR_AREA = 150)** : 어느정도 거리 떨어졌을 때에도 칫솔을 명확히 탐지함
#### 3) 동작 판정 기준
- 최소 유효 속력 **(MIN_BRUSH_SPEED = 2.0 px/frame)** : 일반적으로 양치를 한다고 시뮬레이션 해보았을 때 적정값

<br>

## AI 사용
- 코드 구조 정제 및 문법 검사에 AI 사용
- 모폴로지 연산 하이퍼파라미터를 실시간으로 마우스 슬라이더로 조절하면서 테스트하기 위한 코드 작성에 사용
- 연필 색상 임계값 : 분석 조명 반사와 원통형 음영을 수용하기 위한 권장 LAB 색상 범위 도출
- 타이머 기능 구현
- 화면 상에 보이는 텍스트(얼굴/사물 감지 피드백, 타이머) 구현
- README 파일 내용 문장 다듬기