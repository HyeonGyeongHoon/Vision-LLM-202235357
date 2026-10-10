# Visual Studio Code 가상 환경 생성
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
5. pip 업그레이드:
```shell
python -m pip install --upgrade pip
```
6. Python 인터프리터 설정
- ctrl + shift + p 누른 뒤 Python: Select Interpreter 입력/선택
- 목록에서 ('.venv': venv) 선택

7. 실습 폴더 생성 및 이동
```shell
mkdir -p exercise
cd exercise
```
8. 실습 파일 열기
```shell
code exercise.py
```
9. 실습파일 편집
- VS Code 에디터 창에 코드 붙여넣기 후 Ctrl + S로 저장
10. 파이썬 파일 실행
- 에디터 우측 상단의 재생 버튼 클릭