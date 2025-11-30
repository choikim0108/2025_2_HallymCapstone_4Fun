# 🚀 메타버스 기반 상황극형 외국어 학습 플랫폼 : 4Fun

![Project Logo or Banner](https://github.com/user-attachments/assets/796f941c-dd3a-4f40-b44f-bd66c117c169)

![Unity](https://img.shields.io/badge/Unity-6000.0.28f1-black?style=flat&logo=unity)
![Platform](https://img.shields.io/badge/Platform-PC-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Photon](https://img.shields.io/badge/Network-Photon_PUN2-00B2FF)

본 프로젝트는 Unity 엔진을 기반으로 하는 **메타버스 기반 상황극형 외국어 학습 플랫폼**입니다. 교사 1인과 학생 3인이 실시간으로 상호작용하며 학습 콘텐츠를 체험하는 몰입형 교육 환경을 제공하는 것을 목표로 합니다.

---

## 1. 프로젝트 개요

* **프로젝트 명**: 4Fun
* **개발 기간**: 2025.02 ~ 2025.11
* **핵심 목표**: 게임형 콘텐츠와 AI 힌트 시스템을 결합하여 학습 동기를 부여하고, WhisperX 기반 음성 분석 리포트를 통해 실질적인 학습 성과를 도출합니다.
* **플랫폼**: PC (Windows/Mac), 향후 모바일 및 VR 확장 고려
* **참여 인원**: 최대 4인 (교사 1, 학생 3, 확장 가능)

---

## 2. 주요 기능

### 🎮 Gameplay & Interaction
* **[실시간 멀티플레이]** Photon(PUN 2) 활용, 위치/애니메이션/상호작용 실시간 동기화.
* **[세션 관리]** 교사(Host)가 맵(우체국 등) 생성 및 세션 코드를 통한 학생(Client) 접속.
* **[커스터마이징]** Firebase 연동 아바타 및 인게임 재화 시스템.

### 🤖 AI Integration
* **[AI 보조 교사]** GPT-4o 기반 프롬프트 엔지니어링이 적용된 실시간 힌트 시스템.
* **[음성 분석 리포트]**
    1. 학생 음성 녹음 및 추출
    2. **WhisperX** 모델을 통한 발음/유창성 분석
    3. 정량화된 피드백 리포트 생성
---

## 3. 기술 스택

| 구분 | 기술 | 비고 |
| :--- | :--- | :--- |
| **Game Engine** | `Unity (C#)` | Unity 6.2 (6000.2.8f1) |
| **Networking** | `Photon (PUN 2)` | Unity 기반 실시간 멀티플레이 동기화 |
| **Backend & DB** | `Firebase` | Auth, Database |
| **AI (Hint)** | `gpt-4o` | AI Hint System |
| **AI (Session Analyzer)** | `WhisperX` | 세션 내 음성 분석 및 리포트 생성 |
| **Version Control** | `Unity DevOps (Plastic SCM)` / `Git` | |

---

## 4. 팀 구성 및 역할

| 이름 | 역할 | 담당 업무 | GitHub |
| :---: | :---: | :--- | :---: |
| **김지훈** | **Team Leader** | PM, Unity 클라이언트 핵심 로직, AI(WhisperX) 서버 개발 | [@GitHubID](https://github.com/choikim0108) |
| **원현섭** | **Developer** | Photon 네트워크 동기화, Firebase 백엔드 연동, 매칭 시스템 | [@GitHubID](https://github.com/choikim0108) |
| **김태건** | **Designer/Dev** | UI/UX 기획 및 디자인, 로비/인게임 UI 구현 | [@GitHubID](https://blog.naver.com/noegeat)|

---

## 5. 관련 링크

* **Notion**: [Project 4Fun](https://www.notion.so/4Fun-26708879c77f80a89a66f611c2d174f6)
* **GitHub**: [choikim0108/2025_2_HallymCapstone_4Fun](https://github.com/choikim0108/2025_2_HallymCapstone_4Fun/)
* **Version Control (DevOps)**: [Unity DevOps Dashboard](https://cloud.unity.com/home/organizations/14569676474665/projects/08340abb-5539-4b3a-a144-18a85afa6a18/cloud-build/config)
