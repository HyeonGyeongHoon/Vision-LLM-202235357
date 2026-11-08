# Vision-LLM-202235357

## AI & Embedded Systems Engineering Portfolio

본 저장소는 가천대학교 AI 부트 캠프 수업 `온디바이스AI` 수업에시 진행한 컴퓨터 비전, 딥러닝, 객체 검출, 그리고 최신 LLM 및 멀티모달 시스템에 관한 실습 코드와 종합 프로젝트 결과물을 담고 있는 포트폴리오 저장소입니다.

## 📁 Directory Structure

```text
├── exercise_261010/          # Linux, Python 및 기초 임베디드 실습
├── exercise_261017/          # 컴퓨터 비전 및 OpenCV 기초 실습
├── exercise_261025/          # 딥러닝, GPU 가속 및 이미지 분류 실습
├── Final_Project/            # 최종 종합 프로젝트 (발표 자료 및 소스 코드)
├── Project01/                # 첫 번째 미니 프로젝트 소스 코드
├── src/                      # 공통 리소스 (데이터셋, 모델 가중치, 이미지 및 오디오)
├── *.ipynb                   # 단계별 주피터 노트북 학습 자료
└── *.py                      # 각종 감지(볼, 손가락, FPS), STT/TTS 통합 실습 스크립트
```

---

## 🚀 Key Modules & Contents

### 1. 📚 주피터 노트북 학습 자료 (`*.ipynb`)
* **`01_Linux-and-Python.ipynb`**: 리눅스 환경 설정 및 파이썬 기초 프로그래밍
* **`02_Computer-Vision.ipynb`**: OpenCV를 활용한 이미지 처리 및 컴퓨터 비전 기초
* **`03_DL-and-GPU.ipynb`**: 딥러닝 기초 및 GPU 가속 환경 구성
* **`04_DL-Object-Detection.ipynb`**: YOLO 및 객체 검출 모델 실습
* **`05-LLM_and_Gemma.ipynb`**: 대형 언어 모델(LLM) 구조 이해 및 Gemma 활용 실습
* **`06-Vision_LLM_Multimodal_Systems.ipynb`**: 비전 LLM 및 멀티모달 AI 시스템 통합 실습

### 2. 🛠️ 핵심 파이썬 스크립트 (`*.py`)
* **객체 및 비전 감지**: `ball_detection.py`, `count_finger.py`, `kalman_filter_ball.py`, `media_pipe_test.py`
* **성능 최적화**: `detect_fps.py`, `detect_fps_int8.py` (INT8 양자화 및 FPS 측정)
* **음성 및 멀티모달 연동**: `test_stt.py`, `test_tts.py`, `integrating_sst_and_tts.py`
* **유틸리티 및 하드웨어 연동**: `test_cam.py`, `take_calibration.py`

### 3. 📂 디렉터리별 상세 내용
* **`exercise_261010/` ~ `261025/`**: 날짜별 집중 실습 내용 (GPU 가속, CNN 학습, MNIST 분류, 바운딩 박스 처리 등)
* **`Project01/`**: 핵심 기능 구현 미니 프로젝트 (`main.py`)
* **`Final_Project/`**: 최종 성과물 소스 코드 (`main.py`, `test_llm.py`) 및 프로젝트 발표 자료 (`Final_Project_deck.pdf`)
* **`src/`**: 실습에 사용되는 이미지 (`apple.png`, `seagull.jpg` 등), 캘리브레이션 데이터, YOLO/MediaPipe/CIFAR10/MNIST 학습 모델 가중치 저장소

---
Vision-LLM-202235357 현경훈