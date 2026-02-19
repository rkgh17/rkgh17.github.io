---
title: '[Obsidian] Git 플러그인 "Unchanged lines" 변경 사항 이슈 해결(Windows ↔ Android 동기화)'
date: 2026-02-19 14:11:00 +0900
categories: [Development]
tags: [Obsidian, Git, Troubleshooting]
image:
    path: /assets/img/posts/20260219_1/thumbnail.png
---

어느날부터 옵시디언 Git 플러그인에서 내용을 수정하지 않았음에도 `unchanged lines`라고 뜨며 파일이 **modified** 상태로 잡히는 현상이 발생했다.  
Discard하거나 그냥 무시하고 push&pull해도 동기화 자체에는 큰 문제가 없었지만, 커밋 로그가 지저분해지고 계속 거슬려서 원인을 찾아 해결해보았다.

<br/>

---

![img](/assets/img/posts/20260219_1/content1.png)
_Obsidian 화면_

## **원인**

원인은 **운영체제 간의 줄바꿈 문자(EOL, End Of Line) 처리 방식의 차이** 때문이었다.  
현재 나는 **Android(Termux)**와 **Windows**를 연동하여 옵시디언을 관리하고 있는데, Windows에서 작성된 파일과 Android(Linux기반)에서 처리하는 파일의 줄바꿈 규칙이 서로 충돌이 일어났다.

### EOL(End Of Line)
컴퓨터가 줄바꿈을 인식하는 제어 문자에는 CRLF, LF가 있으며 차이는 다음과 같다.

- **CRLF** (\r\n) : Carriage Return + Line Feed
	- 주로 Windows에서 사용
- **LF** (\n) : Line Feed
	- 주로 Unix 계열(Linux, macOS)에서 사용

캐리지 리턴같은경우는 타자기에서 유래했다는데.. 영화에서 타자기 쓰는거 보면 띵! 소리나면서 줄바꿈(line feed) + 맨 왼쪽으로 리셋(Carriage Return)하는 경우가 있는데 그거라고 한다. (굳이?)

---

## **해결**

Git에게 현 프로젝트에서 사용하는 줄바꿈을 명시해준다.

### **.gitattributes 파일 생성**
루트 디렉토리에서 다음 명령어를 통해 모든 줄바꿈을 LF으로 강제한다는 규칙을 담은 파일을 생성해준다.

```bash
echo "* text=auto eol=lf" > .gitattributes
```

### **Renormalize**

- 해당 규칙에 맞게 파일을 정렬한다.

```bash
git add --renormalize
```

이제 변경사항 커밋 & 푸시하면 된다. 이후부터는 어디서 작업하든 Git이 알아서 LF로 통일하여 처리하므로 unchanged lines 이슈가 사라진다.

---

## **마무리**

사실 개인적인 불편함 때문에 찾아본 정보였지만, CRLF / LF 이슈는 개발 환경에서 꽤 중요한 주제이다.  
Docker환경에서 윈도우에서 작성한 스크립트를 리눅스 컨테이너로 복사해서 실행할 때, CRLF때문에 "command not fouond" 같은 에러가 종종 발생한다.  
앞으로 OS 간의 환경차이가 발생하는 업무(DevOps, 서버 배포 등)를 볼때는 이 EOL 이슈를 항상 염두에 두어야겠다.
