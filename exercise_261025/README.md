# 261025 과제 제출

## 실행 파일
- main.py: 이미지 잔상 효과 구현 코드

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

## 기타 참조 사항
### motion trail 반복문 수정
- GPU 웜업 코드에서 더미 데이터를 처리하기 위해 range(TRAIL_LENGTH)로 정적으로 되어있던 부분을 입력받은 weight의 배열 길이에 따라 동적으로 작동하도록 수정함