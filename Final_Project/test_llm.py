"""
test_llm.py:  main.py의 LLM(Gemma) 응답 생성 검증 스크립트

실제 하드웨어(카메라, 마이크) 없이 다양한 소품 상태와 의도(Intent) 시나리오를 주입하여
LLM에 전송되는 최종 프롬프트 문구와 실제로 생성되는 남친 멘트 응답을 직접 확인하는 테스트 파일입니다.
"""

import sys

# 1.  main.py 모듈 및 LLM 모델 불러오기
try:
    import  main as target_module
except ImportError:
    import main as target_module


def test_llm_responses():
    print("=" * 70)
    print(" 🤖 Gemma LLM 시나리오별 대화 응답 생성 테스트")
    print("=" * 70)

    # 검증할 다양한 소품 및 의도 시나리오 정의
    scenarios = [
        {
            "title": "1. INITIAL_LOOK (착용 소품 있음: 가방, 텀블러)",
            "intent": "INITIAL_LOOK",
            "current_items": {"backpack", "bottle"},
            "added": None,
            "removed": None,
        },
        {
            "title": "2. INITIAL_LOOK (착용 소품 없음)",
            "intent": "INITIAL_LOOK",
            "current_items": set(),
            "added": None,
            "removed": None,
        },
        {
            "title": "3. DIFF_LOOK (새 소품 추가됨: 우산 챙김)",
            "intent": "DIFF_LOOK",
            "current_items": {"backpack", "umbrella"},
            "added": {"umbrella"},
            "removed": set(),
        },
        {
            "title": "4. DIFF_LOOK (기존 소품 내려놓음: 가방 내려놓음)",
            "intent": "DIFF_LOOK",
            "current_items": set(),
            "added": set(),
            "removed": {"backpack"},
        },
        {
            "title": "5. DIFF_LOOK (소품 교체됨: 가방 -> 넥타이)",
            "intent": "DIFF_LOOK",
            "current_items": {"tie"},
            "added": {"tie"},
            "removed": {"backpack"},
        },
        {
            "title": "6. DIFF_LOOK (달라진 소품 없음 - 당황하는 멘트 유도)",
            "intent": "DIFF_LOOK",
            "current_items": {"backpack"},
            "added": set(),
            "removed": set(),
        },
        {
            "title": "7. NO_MEMORY (이전 기억 없음 - 사과 멘트 유도)",
            "intent": "NO_MEMORY",
            "current_items": {"backpack"},
            "added": None,
            "removed": None,
        },
    ]

    for idx, sc in enumerate(scenarios, 1):
        print(f"\n[{sc['title']}]")

        # LLM 응답 생성 함수 호출 ( main.py 내 실제 Gemma 모델 추론)
        reply = target_module.generate_boyfriend_reply(
            intent=sc["intent"],
            current_items=sc["current_items"],
            added=sc["added"],
            removed=sc["removed"],
        )

        print(f"👉 LLM 최종 응답:\n\"{reply}\"")
        print("-" * 70)


if __name__ == "__main__":
    test_llm_responses()