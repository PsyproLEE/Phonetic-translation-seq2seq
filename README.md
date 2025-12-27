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

1. 일본어 문장을 Papago를 사용해 변환하여
  -한국어 발음 표기 (phonetic transcription)
  -한국어 의미 표현 (semantic meaning)
   
2. 이렇게 생성된 (발음, 의미) 쌍을 학습 데이터로 사용한다.

이 과정에서 일본어 문장은 모델의 직접 입력으로 사용되지 않으며,
다단계 번역 파이프라인에서 발생할 수 있는 **오류 누적(cascading error)**을 피하기 위해
발음 → 의미 간의 직접 매핑 구조를 채택하였다.

## Task Definition

최종적으로 정의된 학습 과제는 다음과 같다.

** Seq2Seq: 한국어 발음 → 한국어 의미 **

데이터셋은 다음과 같이 표현된다.

D = { (p_i, m_i) }

-p_i: Papago로 생성된 한국어 발음 표기
-m_i: 해당 발음에 대응하는 한국어 의미 표현

## Example

**Input** (Korean phonetic transcription):  
`아리가또 고자이마스`

**Output** (Korean semantic meaning):  
`감사합니다`

## Project Background

본 프로젝트는 대학 딥러닝 수업에서 진행된 팀 프로젝트를 기반으로 한다.
수업 종료 후 코드 구조 정리와 실험 재현성이 충분히 확보되지 않아,
이를 개인적으로 재구성 및 정리하여 하나의 독립적인 레포지토리로 확장하였다.

본 레포지토리는 모델 구조 정리, 데이터 생성 파이프라인 명확화,
그리고 향후 확장을 위한 기반을 목적으로 한다.
