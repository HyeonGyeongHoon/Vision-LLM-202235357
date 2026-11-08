import os  # 파일 경로 생성
import re  # 정규표현식 기반 키워드 패턴 매칭을 위해 사용
import subprocess  # 외부 C++/독립 실행 파일을 파이썬 내부에서 제어하기 위해 사용
import cv2
from ultralytics import YOLO
from llama_cpp import Llama

# 파일 경로 및 환경 설정
YOLO_MODEL_PATH = "../src/models/YOLO/yolo11n_int8.engine"  # int8 양자화 엔진

GEMMA_MODEL_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"  # Gemma local llm 호출

WHISPER_CLI = "../whisper.cpp/build-cpu/bin/whisper-cli"  # STT
WHISPER_MODEL = "../whisper.cpp/models/ggml-base.bin"

PIPER_PYTHON = "../.piper_venv/bin/python"  # TTS
PIPER_MODEL = "../src/models/Piper/ko_KR-kss-medium.onnx"

AUDIO_INPUT_PATH = "../src/audio/input.wav"
AUDIO_OUTPUT_PATH = "../src/audio/response.wav"
MIC_DEVICE = "plughw:3,0"

PATTERN1_CLASSES = [24, 25, 26, 27, 28, 39, 41]
# "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "39": "bottle", "41": "cup"

ITEM_KOREAN_MAP = {
    'backpack': '가방',
    'umbrella': '우산',
    'handbag': '가방',
    'tie': '넥타이',
    'suitcase': '캐리어',
    'bottle': '텀블러',
    'cup': '컵',
}

# 모델 초기화
print("Loading models...")
yolo = YOLO(YOLO_MODEL_PATH)
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)
print("Models loaded successfully.")

# 오디오 녹음
def record_audio_jetson(sec=4, output_file=AUDIO_INPUT_PATH):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(sec),
            output_file,
        ],
        check=True,
    )
    return output_file

# STT 음성 인식
def run_stt(audio_file=AUDIO_INPUT_PATH):
    result = subprocess.run(
        [
            WHISPER_CLI,
            "-m", WHISPER_MODEL,
            "-f", audio_file,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()

# TTS 음성 출력
def run_tts(text, output_file=AUDIO_OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m", "piper",
            "-m", PIPER_MODEL,
            "-f", output_file,
            "--",
            text,
        ],
        check=True,
    )
    subprocess.run(["aplay", output_file], check=True)

# 소품 감지
def get_detected_objects(frame):
    results = yolo.predict(source=frame, conf=0.25, classes=PATTERN1_CLASSES, verbose=False)
    detected = set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())  # 탐지된 객체의 클래스 ID 추출
        cls_name = results[0].names[cls_id]  # 클래스 이름(문자열)으로 변환
        detected.add(cls_name)
    return detected

# 의도 분류
def classify_intent(user_speech, has_memory=False):
    if not user_speech or not user_speech.strip():  # 음성을 인식하지 못한 경우
        return "UNKNOWN"

    clean_text = user_speech.strip()

    diff_patterns = [
        r"달라", r"바[뀌꿰뀐꿨꾼]", r"차이"
    ]
    initial_patterns = [
        r"어때", r"예뻐", r"이뻐", r"스타일", r"착장", r"패션", r"오늘"
    ]

    is_diff_intent = any(re.search(pattern, clean_text) for pattern in diff_patterns)
    is_initial_intent = any(re.search(pattern, clean_text) for pattern in initial_patterns)

    if is_diff_intent:
        return "DIFF_LOOK" if has_memory else "NO_MEMORY"

    elif is_initial_intent:
        return "INITIAL_LOOK"

    return "UNKNOWN"

# LLM 답변 생성
def generate_boyfriend_reply(intent, current_items=None, added=None, removed=None):
    def format_items(items):
        if not items:
            return None
        korean_names = sorted(list(set([ITEM_KOREAN_MAP.get(item, item) for item in items])))
        return ", ".join(korean_names)

    system_instruction = (
        "너는 여자친구를 너무나 사랑하는 다정하고 센스 있는 남자친구야.\n"
        "[작성 규칙]\n"
        "1. 상황에 맞게 센스 있고 다정하게 칭찬해.\n"
        "2. 엉뚱한 딴소리는 절대 하지 마.\n"
        "3. 다정한 반말 남친 말투로 딱 2문장으로만 대답해."
    )

    user_content = ""
    if intent == "INITIAL_LOOK":
        current_str = format_items(current_items)
        if current_str:
            user_content = f"여자친구 질문: '자기야, 오늘 나 어때?'\n착용 중인 소품: {current_str}\n요청: 소품의 분위기를 짚어서 다정하게 칭찬해줘."
        else:
            user_content = "여자친구 질문: '자기야, 오늘 나 어때?'\n착용 중인 소품: 없음\n요청: 기본 스타일과 분위기가 너무 예쁘다고 칭찬해줘."

    elif intent == "DIFF_LOOK":
        added_str = format_items(added)
        removed_str = format_items(removed)

        if removed_str and not added_str:
            user_content = f"여자친구 질문: '자기야, 나 뭐 달라진 거 없어?'\n변화: 아까 있던 {removed_str}을/를 내려놓았음\n요청: 아까 든 소품 어디 갔냐고 물어보며 예쁘다고 칭찬해줘."
        elif added_str and not removed_str:
            user_content = f"여자친구 질문: '자기야, 나 뭐 달라진 거 없어?'\n변화: 없던 {added_str}을/를 새로 챙김\n요청: 새로 챙긴 소품을 바로 알아채며 센스 있게 칭찬해줘."
        elif added_str and removed_str:
            user_content = f"여자친구 질문: '자기야, 나 뭐 달라진 거 없어?'\n변화: {removed_str} 대신 {added_str}로 바꿈\n요청: 바뀐 소품을 콕 집어 알아채며 칭찬해줘."
        else:
            user_content = "여자친구 질문: '자기야, 나 뭐 달라진 거 없어?'\n변화: 달라진 것 없음\n요청: 바뀐 게 없어서 살짝 당황하면서도 아까랑 똑같이 예쁘다고 말해줘."

    elif intent == "NO_MEMORY":
        user_content = (
            "여자친구 질문: '자기야, 나 뭐 달라진 거 없어?'\n"
            "상황: 이전 모습을 기억하지 못함\n"
            "요청: 아까 모습을 못 봐서 미안하다고 솔직하게 사과하고, 지금 모습이 너무 예쁘다고 칭찬해줘."
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=120,  # 문장 짤림 방지 여유 있게 설정
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"].strip()

# 메인 실행 루프
def main():
    # pipeline 정의
    pipeline = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    prev_items = None
    print("\nControls:")
    print("  's': Start voice recording (4s)")
    print("  'q': Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 's' to Talk, 'q' to Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Boyfriend AI", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('s'):
            # 1. 음성 녹음 수행 (4초간 마이크 입력 수집)
            print("\n[Recording audio...]")
            record_audio_jetson(sec=4)

            # 2. STT 음성 인식 (녹음된 오디오를 텍스트로 변환)
            print("[Running STT...]")
            user_speech = run_stt(AUDIO_INPUT_PATH)
            print(f"User speech: \"{user_speech}\"")

            # 3. 의도 분류 및 입력 상태 확인 (이전 기억 유무 판별 후 Intent 분류)
            has_memory = (prev_items is not None)
            intent = classify_intent(user_speech, has_memory=has_memory)

            # UNKNOWN 예외 처리: 음성 미인식 시 YOLO 추론을 진행하지 않고 즉시 응답 후 조기 리턴(continue)
            if intent == "UNKNOWN":
                print("[Warning] Intent could not be determined.")
                reply = "자기야 방금 뭐라고 했어? 목소리가 잘 안 들렸는데 너무 예뻐서 딴생각했나 봐!"
                print(f"Response: \"{reply}\"")
                run_tts(reply)
                continue

            # 4. 카메라 프레임 기반 소품 감지 (YOLO 추론 실행)
            current_items = get_detected_objects(frame)

            # 5. 의도에 따른 소품 변화 비교 및 기억 갱신
            if intent == "INITIAL_LOOK":
                added, removed = None, None
                prev_items = current_items.copy()
            elif intent == "DIFF_LOOK":
                added = current_items - prev_items
                removed = prev_items - current_items
                prev_items = current_items.copy()
            elif intent == "NO_MEMORY":
                added, removed = None, None
                prev_items = current_items.copy()

            # 6. LLM 응답 생성 및 TTS 음성 출력
            print("[Generating LLM response...]")
            reply = generate_boyfriend_reply(
                intent=intent,
                current_items=current_items,
                added=added,
                removed=removed
            )
            print(f"Response: \"{reply}\"")

            print("[Running TTS...]")
            run_tts(reply)
            print("Done.\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()