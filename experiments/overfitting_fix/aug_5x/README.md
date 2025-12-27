# Overfitting Mitigation via Multi-level Data Augmentation (x5)

본 실험은 기본 데이터셋에서 관찰된 과적합(overfitting) 현상을 완화하기 위해,
발음 기반 노이즈 증강을 다중 수준으로 적용한 데이터 확장 실험이다.

## Motivation

기존 baseline 실험에서는,
- 학습 BLEU는 빠르게 상승하나
- validation BLEU가 일정 epoch 이후 정체 또는 하락하는 현상이 관찰되었다.

이는 입력 발음 표기가 상대적으로 규칙적이고
데이터 다양성이 부족한 데서 기인한 것으로 판단하였다.

## Augmentation Strategy

원본 데이터 1배에 대해,
서로 다른 노이즈 전략을 적용한 4종의 증강 데이터를 추가하여
총 **x5 규모의 학습 데이터**를 구성하였다.

### Applied Noise Types

- Pronunciation noise (prob = 0.3)
- Pronunciation noise (prob = 0.7)
- Random drop noise  
  (일부 음절 또는 글자 제거)
- Random change noise  
  (유사 발음 간 임의 치환)

모든 노이즈는 **입력(발음) 시퀀스에만 적용**되며,
출력(의미) 문장은 원본을 유지한다.

## Dataset Composition

Total dataset = Original
+ aug(prob=0.3)
+ aug(prob=0.7)
+ random drop
+ random change


## Training Setup

- 동일한 Seq2Seq with Attention 모델 구조 사용
- 동일한 optimizer 및 learning rate 유지
- 증강 효과만을 비교하기 위해
  모델 구조 및 학습 전략은 baseline과 동일하게 설정

세부 하이퍼파라미터는 `config.yaml`을 참고한다.

## Expected Effect

본 실험의 목적은 단순 BLEU 점수의 최대화가 아니라,
- validation 성능의 안정화
- 일반화 성능 향상
- 과도한 memorization 감소

에 있다.

