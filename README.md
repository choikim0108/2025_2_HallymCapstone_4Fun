# 🚀 AI Tutor 기반 교육용 멀티플레이 게임: Project 4Fun

본 프로젝트는 Unity 엔진을 기반으로 하는 **AI 튜터 연동형 교육 메타버스**입니다. 교사 1인과 학생 3인이 실시간으로 상호작용하며 학습 콘텐츠를 체험하는 몰입형 교육 환경을 제공하는 것을 목표로 합니다.

---

## 1. 프로젝트 개요

* **프로젝트 명**: 4Fun (가제)
* **핵심 목표**: 게임형 콘텐츠와 AI 보조 교사(AI Tutor)를 결합하여 학습 동기를 부여하고, 음성 분석 리포트를 통해 실질적인 학습 성과를 도출합니다.
* **플랫폼**: PC (Windows/Mac), 향후 모바일 및 VR 확장 고려
* **참여 인원**: 최대 4인 (교사 1, 학생 3)

---

## 2. 주요 기능 (Core Features)

* **[세션 관리]** 교사(Host)가 학습 맵(예: 공항 시뮬레이션)을 생성하고, 학생(Client)들은 고유 세션 코드를 통해 참여합니다.
* **[AI 보조 교사]** 프롬프트 엔지니어링이 적용된 LLM(대형 언어 모델) 기반의 AI 튜터가 학생들의 학습을 실시간으로 보조합니다.
* **[음성 인식 및 분석]** 학생의 음성을 녹음/추출하고, Python 기반 AI 모델로 발음 및 유창성을 분석하여 정량화된 **학습 리포트**를 생성합니다.
* **[실시간 멀티플레이]** Photon(PUN 2)을 활용하여 플레이어 간 위치, 애니메이션, 상호작용을 실시간 동기화합니다.
* **[데이터 관리]** Firebase (Firebase Auth, Firebase Database)를 사용하여 사용자 인증, 프로필, 커스터마이징 데이터를 관리합니다.
* **[플레이어 커스터마이징]** Firebase와 연동된 아바타 및 프로필 커스터마이징 기능을 제공합니다.

---

## 3. 기술 스택 (Tech Stack)

| 구분 | 기술 | 비고 |
| :--- | :--- | :--- |
| **Game Engine** | `Unity (C#)` | Unity 6.2 (6000.2.8f1) |
| **Networking** | `Photon (PUN 2)` | Unity 기반 실시간 멀티플레이 동기화 |
| **Backend & DB** | `Firebase` | Auth, Database |
| **AI (Tutor)** | `LLM API` | (Gemini / GPT 등) |
| **AI (Voice)** | `Python` | 음성 분석 및 리포트 생성 (e.g., Librosa, STT) |
| **Version Control** | `Unity DevOps (Plastic SCM)` / `Git` | |

---

## 4. 팀 구성 및 역할 (R&R)

### 🧑‍💻 김지훈 (Project Lead / Full-Stack Developer)
* [ ] 프로젝트 총괄 및 아키텍처 설계
* [ ] Unity 클라이언트 핵심 로직 개발 (상호작용, AI 연동 등)
* [ ] Python 기반 음성 분석 AI 모델 연구 및 개발

### 🧑‍💻 원현섭 (Network & Backend Developer)
* [ ] 멀티플레이 네트워크 솔루션 (Photon) 리서치 및 구축
* [ ] Firebase (Auth, DB) 연동 및 백엔드 환경 설정
* [ ] 세션 매칭 및 실시간 동기화 로직 구현

### 🧑‍💻 김태건 (UI/UX Designer & Frontend Developer)
* [ ] 게임 전체 UI/UX 기획 및 와이어프레임 디자인
* [ ] 로비, 인게임 UI 및 사용자 상호작용(UX) 구현
* [ ] 플레이어 커스터마이징 UI/UX 개발

---

## 5. 관련 링크 (Links)

* **Notion (WIP)**: [Project 4Fun Wiki](https://www.notion.so/4Fun-26708879c77f80a89a66f611c2d174f6)
* **GitHub**: [choikim0108/2025_2_HallymCapstone_4Fun](https://github.com/choikim0108/2025_2_HallymCapstone_4Fun/)
* **Version Control (DevOps)**: [Unity DevOps Dashboard](https://cloud.unity.com/home/organizations/14569676474665/projects/08340abb-5539-4b3a-a144-18a85afa6a18/cloud-build/config)
