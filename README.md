## Overview

본 프로젝트는 한국어 발음 기반 입력을 한국어 의미 표현으로 직접 변환하는
Attention 기반 Seq2Seq 모델을 제안한다.

핵심 아이디어는, 외국어 문장을 들은 화자가 이를 자신의 언어 체계에 맞는 발음 표기로 인식한다는 점에 착안하여,
발음(phonetic transcription)이라는 비정형·노이즈 입력으로부터 의미를 복원하는 문제를
end-to-end sequence-to-sequence task로 정의한 것이다.

본 모델의 최종 입력과 출력은 모두 한국어이며,
일본어 문장은 데이터셋 생성을 위한 소스 코퍼스로만 사용된다.


## Dataset Construction

데이터셋은 다음 절차를 통해 구성되었다.

1. 일본어 문장을 Papago를 사용해 변환하여<br>
    - 한국어 발음 표기 (phonetic transcription)<br>
    - 한국어 의미 표현 (semantic meaning)
   
2. 이렇게 생성된 (발음, 의미) 쌍을 학습 데이터로 사용한다.

이 과정에서 일본어 문장은 모델의 직접 입력으로 사용되지 않으며,<br>
다단계 번역 파이프라인에서 발생할 수 있는 **오류 누적(cascading error)**을 피하기 위해
발음 → 의미 간의 직접 매핑 구조를 채택하였다.

## Task Definition

최종적으로 정의된 학습 과제는 다음과 같다.

**Seq2Seq: 한국어 발음 → 한국어 의미**

데이터셋은 다음과 같이 표현된다.

D = { (p_i, m_i) }

p_i: Papago로 생성된 한국어 발음 표기<br>
m_i: 해당 발음에 대응하는 한국어 의미 표현

## Example

**Input** (Korean phonetic transcription):  
`아리가또 고자이마스`

**Output** (Korean semantic meaning):  
`감사합니다`

## Repository Structure

본 레포지토리는 데이터 구축, 증강, 실험, 모델 구현 단계를 명확히 분리하여
연구 지향적인 워크플로우를 따르도록 구성되어 있다.

```text
Phonetic-translation-seq2seq/
│
├── data/
│   ├── raw/                    # 원본 일본어 문장 (비공개)
│   ├── augmentation/           # 발음 노이즈 증강 스크립트
│   │   ├── augment_dataset.py
│   │   └── README.md
│   ├── make_splits.py          # train / val / test 분할 스크립트
│   └── README.md               # 데이터 구성 및 생성 과정 설명
│
├── experiments/
│   ├── baseline/               # 증강 강도별 기본 실험
│   │   ├── aug_0.3/
│   │   ├── aug_0.5/
│   │   ├── aug_0.7/
│   │   └── aug_1.0/
│   │
│   ├── overfitting_fix/aug_5x/ # 데이터 x5 증강 기반 과적합 완화 실험
│   │   ├── config.yaml
│   │   └── README.md
│   │
│   └── README.md               # 실험 설정 및 BLEU 평가 결과 정리
│
├── models/
│   └── seq2seq/
│       ├── seq2seq.py          # Seq2Seq with Attention 모델 정의
│       ├── train.py            # 학습 스크립트
│       ├── eval.py             # BLEU 기반 평가 스크립트
│       └── infer.py            # 추론(inference) 스크립트
│
├── config/
│   └── seq2seq.yaml            # 최종 실험 파라미터 설정
│
├── utils/
│   ├── __init__.py      # 공용 유틸리티 패키지 초기화
│   ├── config.py        # YAML 기반 실험 설정 로딩 및 파라미터 관리
|   ├── dataset.py       # Seq2Seq 학습을 위한 Dataset / DataLoader 정의
|   └── vocab.py         # 문자 단위 vocab 생성 및 직렬화 (src / tgt)
|
├── docs/                       # 발표 자료 및 문서
│
├── README.md                   # 프로젝트 개요 및 전체 설명
└── requirements.txt            # 실험 환경 재현을 위한 의존성 목록

```
## Project Background

본 프로젝트는 대학 딥러닝 수업에서 진행된 팀 프로젝트를 기반으로 한다.
수업 종료 후 코드 구조 정리와 실험 재현성이 충분히 확보되지 않아,
이를 개인적으로 재구성 및 정리하여 하나의 독립적인 레포지토리로 확장하였다.

본 레포지토리는 모델 구조 정리, 데이터 생성 파이프라인 명확화,
그리고 향후 확장을 위한 기반을 목적으로 한다.
