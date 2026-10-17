# 261010 과제 제출

## 실행 파일
- exercise.py: Todo 과제 수행 파일
- section_project.py: Section Project 과제 수행 파일

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
```shell
code section_project.py
```
- 또는 New File.. 눌러서 파일 생성
6. 실습파일 편집
- VS Code 에디터 창에 코드 붙여넣기 후 Ctrl + S로 저장
7. 파이썬 파일 실행
- 에디터 우측 상단의 재생 버튼 클릭