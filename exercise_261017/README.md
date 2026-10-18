# 261017 과제 제출

## 실행 파일
- exercise.py: 농구공의 외곽선 검출 코드

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
code exercise.py
```
- 또는 New File.. 눌러서 파일 생성
6. 실습파일 편집
- VS Code 에디터 창에 코드 붙여넣기 후 Ctrl + S로 저장
7. 파이썬 파일 실행
- 에디터 우측 상단의 재생 버튼 클릭

<br>

## 기타 참조 사항
### 커널 구조 설계
- 원형인 농구공의 외곽 윤곽선을 자연스럽게 보존하기 위해서 타원형 커널(`cv2.MORPH_ELLIPSE`)를 사용함
### 모폴로지 연산 목적
- 열림을 1번, 2번, 3번 연산 결과, 3번 진행했을 때 윤곽선 추출 대상인 농구공에 맞게 둥근 영역을 가짐
- 닫힘을 1번, 2번, 3번 연산 결과, 큰 차이가 없어 1번만 진행함
### AI 사용
- 코드 구조 정제 및 문법 검사에 AI 사용