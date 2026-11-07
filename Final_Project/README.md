# Final_Project 제출

## 프로젝트 설명

### 1. 개요
- NVIDIA Jetson 온디바이스(On-Device) 환경에서 실시간 영상 및 음성 입력을 기반으로 사용자의 시각적 변화(소품 착용 및 교체)를 감지하고 다정하게 대화하는 멀티모달 남자친구 AI 시스템
- 외부 Cloud 서버 연동 없이 경량화 객체 탐지 모델(YOLOv11 TensorRT), 로컬 LLM(Gemma-4 GGUF), 음성 인식/합성(Whisper.cpp, Piper TTS)을 젯슨 디바이스 내부 GPU로 오프로딩하여 최적화된 대화 반응 제공

### 2. 주요 기능 및 파이프라인
1. **마이크 음성 입력 및 Whisper STT 추론**
    - ALSA CLI(`arecord`)를 이용하여 4초간 사용자 음성 데이터(`input.wav`) 수집
    - C++ 기반 경량 STT 툴인 `whisper-cli`를 파이썬 `subprocess`로 제어하여 음성을 텍스트로 고속 변환
2. **정규표현식(Regex) 기반 Intent 분류 및 조기 파이프라인 파동 최적화**
    - 변환된 텍스트와 이전 대화 기억 유무(`has_memory`)에 따라 의도(`INITIAL_LOOK`, `DIFF_LOOK`, `NO_MEMORY`, `UNKNOWN`)를 분류
3. **YOLOv11 TensorRT(INT8) 기반 소품 객체 탐지**
    - OpenCV로 캡처한 웹캠 프레임 상에서 패션 소품(가방, 우산, 넥타이, 컵 등 7개 클래스) 영역 실시간 추론
    - TensorRT Engine (`yolo11n_int8.engine`) 변환 및 INT8 양자화를 적용하여 Jetson 디바이스 GPU 가속 극대화


4. **차집합 연산 기반 시각 정보 변화 추출**
    - 현재 프레임 감지 집합(`current_items`)과 이전 기억 프레임 집합(`prev_items`) 간의 파이썬 집합 차집합 연산 수행
    - 새로 추가된 소품(`added`), 제거된 소품(`removed`)의 구체적 변화량을 정밀 추출


5. **Gemma LLM 페르소나 대화 생성 및 Piper TTS 음성 출력**
- `llama_cpp`를 활용하여 Gemma GGUF 모델을 전량 GPU 오프로딩(`n_gpu_layers=-1`)하여 칭찬/감상 멘트 생성
- `format_items` 보조 함수를 통해 영문 단어를 한국어로 변환하고 가나다순 정렬(`sorted`)을 적용하여 프롬프트 순서 일관성 확보
- 합성된 응답 텍스트를 `Piper TTS` 및 `aplay`를 통해 스피커로 출력

<br>

## 실행 파일

- main.py: 메인 통합 파이프라인 실행 파일
- test_llm.py: 실제 카메라/마이크 입력 없이 시나리오별 소품 상태 데이터를 주입하여 Gemma LLM 대화 응답 생성을 독립적으로 검증하는 테스트 스크립트 파일

<br>

## Visual Studio Code 가상 환경 생성 및 실행 절차

0. VS Code 좌측 상단 File -> Open Folder에서 프로젝트 폴더 열기
1. Visual Studio Code에서 `Ctrl + Shift + ~` 눌러 터미널 열기
2. `python3-venv` 패키지 설치:

```shell
sudo apt update && sudo apt install python3-venv -y

```

3. 가상 환경 생성 (System Site Packages 포함):

```shell
python3 -m venv --system-site-packages .venv

```

4. 가상 환경 활성화:

```shell
source .venv/bin/activate

```

5. 필요한 패키지 설치:

```shell
pip install opencv-python ultralytics llama-cpp-python

```

6. 실행 파일 확인 및 저장:

```shell
code main.py

```

7. 파이썬 파일 실행:

```shell
python3 main.py

```

- 키보드 제어: `'s'` 키를 눌러 4초간 음성 대화 시작, `'q'` 키를 눌러 프로그램 종료

<br>

## 개발 및 실행 환경 (Dependencies)

- 하드웨어 : NVIDIA Jetson (JetPack 5.x 이상, USB 웹캠 및 마이크/스피커 연동)
- 주요 설치 라이브러리:
- `ultralytics` (YOLOv11 TensorRT 엔진 추론)
- `llama-cpp-python` (Gemma LLM GGUF GPU 오프로딩 추론)
- `opencv-python` (카메라 프레임 입출력 및 화면 렌더링)

 **사전 준비 모델 및 도구 경로**:
- YOLO TensorRT 엔진: `src/models/YOLO/yolo11n_int8.engine`
- Gemma LLM GGUF: `src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf`
- Whisper CLI 및 모델: `whisper.cpp/build-cpu/bin/whisper-cli`, `whisper.cpp/models/ggml-base.bin`
- Piper TTS 및 모델: `.piper_venv/bin/python`, `src/models/Piper/ko_KR-kss-medium.onnx`

<br>

## 기타 참조 사항

### 1. 주요 하이퍼파라미터 및 구조 설정 근거

#### 1) YOLO 객체 탐지 파라미터 (`conf=0.25, INT8 Quantization`)

- **선정 이유**: 오탐을 줄이면서도 실내 조명에서 소품을 신속하게 포착하기 위한 최소 신뢰도 임계값
- **INT8 양자화**: Jetson 하드웨어의 저전력 온디바이스 특성에 맞춰 메모리 점유율을 줄이고 FPS 추론 속도 확보

#### 2) 의도 분류 패턴 매칭 (`re.search` 정규표현식)

- **`diff_patterns` / `initial_patterns**`: "달라", "바뀐", "어때", "예뻐" 등 유연한 어미 변화 대응
- **선정 이유**: 복잡한 NLP 모델 추가에 따른 시스템 딜레이를 방지하고, C-CLI 도구 및 LLM 호출 이전 상단에서 0.1ms 이내로 유휴 판단 수행

#### 3) LLM 추론 파라미터 (`max_tokens=120, temperature=0.3`)

- `max_tokens=120` : 한국어 토큰화 특성(글자당 약 1.5~2.5 토큰) 상 약 50글자 내외의 2문장 답변이 잘리지 않도록 안전 버퍼 확보
- `temperature=0.3` : 다정하고 자연스러운 반말 톤을 유지하면서도 프롬프트 지시사항(2문장 제한, 딴소리 금지)을 벗어나지 않도록 도출 값 고정

### 2. 보완해야 할 점

#### 1) 소품 이외의 의상(옷) 전체 색상/스타일 인지 한계

- 현재 구조는 COCO 80종 중 소품 클래스(가방, 우산, 넥타이 등)가 적음.
- 상/하의 전체 디자인이나 색상 변화를 세밀하게 추적하는 데 한계가 있음.
- 추후 픽셀 단위 딥러닝 시그멘테이션(YOLO-Seg) 또는 패션 특화 분류 모델을 결합하여 보완 예정.

#### 2) 메모리(RAM) 상의 단기 기억의 한계

- 연인 간 대화라는 아이디어와 다르게 여자친구 소품에 대한 정보를 따로 파일로 저장하지 않아, 바로 이전의 데이터만 참고한다는 점에서 한계가 있음 
- 대화 종료 시 prev_items를 memory.json 파일로 저장하여 재시작 후에도 장기 기억 유지

## AI 사용

- Jetson 엣지 디바이스 환경에 최적화된 온디바이스 AI 파이프라인 통합 및 파이썬 코드 리팩토링
- 정규표현식 기반 의도 분류(Intent Classifier) 로직 구성 지원
- README 구조화 및 문서 전처리 정돈 지원
- 카메라/마이크 없이 시나리오별 LLM 대화 출력을 검증하는 test_llm.py 독립 테스트 스크립트 작성에 AI 사용