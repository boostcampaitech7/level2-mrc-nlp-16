# :rocket: Open Domain Question Answering (ODQA)

## :closed_book: 프로젝트 개요
본 프로젝트는 Retrieval을 기반으로한 Question Answering(QA)를 주제로 하고 있습니다. 질문-문맥-정답 쌍으로 구성된 데이터를 활용해서 특정 질문에 대한 정답을 반환하는 모델을 개발하는 것이 목적입니다. 이러한 모델은 방대한 지식에 접근할 수 있는 효율적인 방안을 제시할 수 있습니다. 다양한 표현에 대한 retrieval 성능 개선, reader를 통한 유연한 답변 생성 등 모델 고도화를 통해 기존의 검색 엔진에서 더욱 발전된 검색 기술을 구현할 수 있을 것입니다.

## :family_man_man_boy_boy: 멤버 소개
|강경준|김재겸|원호영|유선우|
|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/da281cf3-b2cc-4022-ae9e-68ed6c174cd7" alt="KKJ" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/5ecd2475-eb8c-4662-bed0-59ae27cc2e0c" alt="KJK" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/e02832ae-f1fa-4b4f-a7e9-fcc511c727e7" alt="WHY" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/d58b7d68-ef8e-4a12-ae6e-0d50458e0c5b" alt="YSW" width="1000" height="250"> |

## :balance_scale: 역할 분담
|팀원| 역할 |
|:---:| --- |
| 강경준 | EDA, 모델링, 모델 실험 코드 관리, 모델 성능 개선 실험 |
| 김재겸 | EDA 및 데이터 검수, 데이터 전처리 및 증강 실험, 모델 서치 및 파라미터 튜닝 등 모델 성능 개발, 앙상블 |
| 원호영 | EDA 및 데이터 검수, 데이터 증강 조사⋅실험 및 관련 코드관리, 모델 서치 및 실험 |
| 유선우 | EDA 및 데이터 검수, 텍스트 정제, 프로젝트 구조 관리, 모델 실험 및 파라미터 튜닝 |

## :computer: 개발/협업 환경
- 컴퓨팅 환경
	- V100 서버 (VS code와 SSH로 연결하여 사용)
- 협업 환경
  	- ![notion](https://img.shields.io/badge/Notion-FFFFFF?style=flat-square&logo=Notion&logoColor=black) ![github](https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white) ![WandB](https://img.shields.io/badge/WeightsandBiases-FFBE00?style=flat-square&logo=WeightsandBiases&logoColor=white)
- 의사소통
  	- ![zoom](https://img.shields.io/badge/Zoom-0B5CFF?style=flat-square&logo=Zoom&logoColor=white)

## :bookmark_tabs: 데이터 설명
- 데이터 구성
  - train_dataset / test_dataset
	  - question: 질문 text
    - context: 질문에 대한 답이 포함된 passage
    - answer (train_dataset only)
      - answer index: context 내에서 정답이 시작되는 index
      - answer text: text형태로 제시된 context 내의 정답
    - wikipedia_documents
      - 위키피디아 문서 집합

## :card_index_dividers: 프로젝트 구조
```
. level2-mrc-nlp-16
├─ .gitihub
├─ data
│  ├─ embedding
│  │  ├─ context_sparse_embedding.bin
│  │  └─ context_dense_embedding.bin
│  ├─ test_dataset
│  └─ train_dataset
├─ data_modules
│  ├─ data_sets.py
│  └─data_loaders.py
├─ model
│  ├─ loss.py
│  ├─ metric.py
│  └─ model.py
├─ utils
│  ├─ __init__.py
│  ├─ add_data.py
│  ├─ embedding.py
│  ├─ augmentation.py
│  ├─ augmentation_requirements.py
│  └─ util.py
├─ .flake8
├─ .gitignore
├─ .gitmessage.txt
├─ .pre-commit-config.yaml
├─ README.md
├─ config_reader.yaml
├─ config_retrieval.yaml
├─ context_dense_embedding.yaml
├─ context_sparse_embedding.yaml
├─ inference.py
├─ requirements.txt
├─ train_reader.py
├─ train_retrieval.py
└─ test.py
```

## :book: 프로젝트 수행 결과
- EDA
	- Unknown token 분석
		- 정상적인 단어임에도, 인식되지 않는 경우
      - 예시
        - 없앴다는
        - 보살핌으로
        - 꾸밈이
        - 옻칠, 옻나무
        - 쨍그렁거릴
        - 슬펐다
      - 예상 처리 방안
        - 인식되지 않는 글자 분석 후 추가
        - 적절한 의미 단위를 토큰으로 추가
      - 예상 효과
        - 더욱 다양한 단어에 대한 인식 가능
        - 의미 단위를 토큰으로 추가하는 경우, 같은 의미를 갖는 단어의 활용형에 대해 다른 방식으로 토큰화가 되는 경우가 생길 수 있을 것으로 예상
    - 외국인 이름을 한글로 표기시, 한글에서 잘 사용되지 않는 글자가 포함되는 경우
      - 예시
        - 벵골
        - 먀스코프스키
        - 듄
        - 베이욘
      - 예상 처리 방안
        - 인식되지 않는 글자 분석 후 추가
      - 예상 효과
        - 질문에 특정 인물의 이름이 직접적으로 들어가는 경우도 많기 때문에, 성능 개발에 도움이 될 것으로 예상
  - Annotation Bias
    - 토큰화된 텍스트를 기준으로 question의 토큰이 context에 포함되는 비율을 통해 annotation bias 측정
    - Summary for covering ratio
      | Statistic | Value |
      |-----------|-------|
      | Mean | 0.70 |
      | Standard deviation | 0.11 |
      | Minimum  | 0.10 |
      | Maximum  | 1.00 |
      - 꽤 높은 covering ratio를 보여줌
      - sparse embedding이 좋은 성능을 보일 것으로 생각됨
    - 예상 처리 방안
      - 유사어 대체 등의 augmentation을 통한 언어 표현의 다양성 확보

- Augmentation
  - 어순 변경(EDA)
    - 설명
      - 임의로 문장의 단어 순서를 변경하여 데이터의 다양성 확보
    - 적용 예시
	    - 대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은? → 대통령을 포함한 미국의 행정부 견제권을 갖는 기관은? 국가
  	  - 현대적 인사조직관리의 시발점이 된 책은? → 현대적 인사조직관리의 시발점이 책은? 된
   	  - 강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가? → 강희제가 글은 쓴 1717년에 누구를 위해 쓰여졌는가?
    - 결과
	    - 증강된 문장과 원래 문장간의 의미 차이는 크지 않음
  	  - 해당 데이터 적용시 성능 저하 발생
   	  - 언어 표현의 다양성 확보 차원에서도 의미 없음
      - | Method | EM | F1 |
        |--------|----|----|
        | Base | 0.5708 | 0.6629 |
        | Augmented | 0.5541 | 0.6385 |
  - 특수 기호 추가(AEDA)
    - 설명
      - text에 임의로 구두점('.', ',', '!', '?', ';')을 추가 하여 데이터의 다양성을 확보
    - 적용 예시
	    - 대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은? → 대통령을 포함한 미국의 행정부 견제권을 . 갖는 국가 기관은? ;
  	  - 현대적 인사조직관리의 시발점이 된 책은? → "현대적 인사조직관리의 시발점이 된 , 책은? ;
   	  - 강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가? → "강희제가 , 1717년에 쓴 글은 , 누구를 위해 쓰여졌는가? ?
    - 결과
	    - EM은 소폭 증가했으나, F1은 소폭 감소
  	  - 유의미한 차이를 발견할 수 없음
      - | Method | EM | F1 |
        |--------|----|----|
        | Base | 0.5708 | 0.6629 |
        | Augmented | 0.5875 | 0.6599 |
  - 질문 생성
    - 설명
      - 주어진 데이터 상의 질문에 이어지는 내용을 생성하여 증강
      - 언어 모델 (skt/kogpt2-base-v2) 기반으로 생성
    - 적용 예시
	    - 대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은? → 대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?? 이런 문제를 해결해달라는 게 아니라 미국의 경제력을 어떻게 키워야 할 것인가? 이런 문제의
  	  - 현대적 인사조직관리의 시발점이 된 책은? → 현대적 인사조직관리의 시발점이 된 책은?이다. 이런 책은 '외부에 의한 조직관리가 아니라 내부의 자발적 조직관리가 이루어져야 한다'는 것을
   	  - 강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가? → 강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가?라는 질문으로 시작되었다. 아니면 누구를 위해 쓰여졌는가? 누가 누구에게
    - 결과
	    - 생성된 문장의 질이 안 좋음
      - | Method | EM | F1 |
        |--------|----|----|
        | Base | 0.5708 | 0.6629 |
        | Augmented | 0.5875 | 0.6772 |
  - 역번역
    - 설명
      - 한국어를 영어로 번역한 뒤 다시 한국어로 번역하는 과정을 통해 text의 다양성 확보
    - 적용 예시
	    - 대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은? → 어떤 국가기관이 대통령을 포함한 미국 행정부를 견제할 권리가 있는가?
  	  - 현대적 인사조직관리의 시발점이 된 책은? → 현대 인사 운영의 출발점이 어떤 책이 됐을까?
   	  - 강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가? → 강희제가 누구를 위해 1717년에 썼나요?
    - 결과
	    - 생성된 문장의 질이 좋음
  	  - 언어 표현의 다양성을 확보하는 데 도움이 됨
      - | Method | EM | F1 |
        |--------|----|----|
        | Base | 0.5708 | 0.6629 |
        | Augmented | 0.5678 | 0.6682 |
	
- Modeling
  - chunking
    - task 설명
      - context가 너무 길어서, 모델의 input size 제한을 넘는 경우 발생
      - 긴 text를 chunk별로 나누어 결과 도출
    - chunking 적용 방안
      - fixed length
        - 일정 길이 단위로 chunking
        - stride를 설정하여 전 후의 chunk가 일정 길이의 공통된 부분 보유
      - truncation
        - 일정 길이로 절단하여 retrieval 진행 (첫 번째 chunk만 활용)
        - 좋은 성능 보임
        - 많은 문서들이 두괄식으로 작성돼 있어서 성능이 괜찮을 수 있다는 의견
      - summary
        - 두괄식 문서를 기대하는 것 보다, summary를 직접 생성 후 활용하는 방안 가능
        - summary를 시도했으나, 즉각적인 성능 향상이 나타나지는 않음
        - 문제 해결을 위한 분석 필요 했지만, 전체 context dataset에 대한 summary 생성에 지나치게 많은 시간 소비되므로 방법론 적용이 어려움
      - chunk별 결과 합산 방안
        - mean
          - 각 text에 대한 chunk 별 embedding vector의 평균을 구하는 방식
        - max
          - 각 chunk별 embedding과 question embedding에 대한  similarity의 최대값 활용
          - 이러한 방식은 search를 위해서 chunk별 embedding을 모두 저장해야 함 → 메모리 문제 발생
      - 문제점
        - chunk 길이가 매우 큰 경우
          - 문제 상황
            - token length 512 기준으로 최대 80개 가량의 chunk가 생성되는 경우까지 존재 → OOM 발생
            - chunk를 나누고 batch size를 1로 만들어도 OOM이 발생
            - reader model에서는 모든 context를 활용해서 답을 찾아야 하기 때문에, 문서를 일정 길이에서 절단 불가
          - 해결 방안
            - 이러한 문제를 해결하기 위해, batch size를 1로 고정하고 각 청크를 개별적으로 모델에 입력하는 방식을 통해, OOM 문제를 피함
            - OOM을 피하기 위해 train data를 한 번에 전부 계산하지 않고, mini-batch를 활용하는 것과 같은 원리
  - hybrid retrieval
    - task 설명
      - question과 context 사이의 단어 표현에 대한 높은 covering ratio 기반으로 sparse embedding의 높은 성능 예상
      - 적절한 augmentation과 추가 데이터 활용이 가능하다면, dense embedding을 활용하여 일반화 성능 향상 가능
    - sparse embedding 적용
      - context별로 길이의 차이가 크기 때문에 이러한 점을 반영하기 위해 bm25 활용
      - parameter test
        | K1 | top-k match ratio |
        |----|-------------------|
        | 0.5 | 0.8833 |
        | 0.8 | 0.9 |
        | 1 | 0.8958 |
        | 2 | 0.8833 |
        | 3 | 0.8708 |
        - K1 : bm25 parameter
        - top-k match ratio : 선택한 k개의 context 중 real context가 포함되는 비율
  - concat retrieval
    - task 설명
      - 두 텍스트를 concat하여 모델 output으로 유사도를 반환하는 방식 적용
      - question과 context간의 attention 활용이 가능하기 때문에, 더 좋은 성능을 보일 것으로 예상
    - 문제점
      - 각각 embedding하는 경우는 search할 때 미리 context에 대한 embedding을 계산한 뒤에, search를 진행하는 것이 가능
      - 하지만, concat method는 새로운 질문이 나올 때 마다 모든 context와 concat을 통한 계산 필요 → 메모리 및 계산 시간 문제
    - 활용 방안
      - Reranking을 활용하여 sparse embedding을 통해 k1개의 문서를 선택한 뒤, 해당 문서에 대해서만 concat retrieval 적용을 통해 계산량 최소화
    - 결과
      | Method | top-k match ratio |
      |--------|-------------------|
      | Not Concat | 0.7625 |
      | Concat | 0.8875 |
      - top-k match ratio : 선택한 k개의 context 중 real context가 포함되는 비율
      - not concat은 기존에 활용하던 sparse embedding과의 weighted mean 방식을 활용하여, 최종 선택까지 sparse embedding의 영향을 받아 더욱 높은 점수가 나오는 것으로 예상
- 모델서치
  - 고려 사항
    - retrieval의 경우 embedding을 생성하는 문제이기 때문에, encoder model 위주로 search
    - reader의 경우 extraction based MRC를 진행할 것이기 때문에 reader model 또한 encoder model 위주로 search
    - 긴 text 처리해야 하는 상황을 고려하여 RoBERTa 계열 모델 위주로 활용
      - BERT는 아무 두 문장을 붙여서 학습하기 때문에, 아주 짧은 케이스도 존재
      - RoBERTa는 토큰화된 문장 길이가 512가 넘지 않는 선에서 최대한 문장을 이어 붙여서 학습
      - 따라서, 여러 문장으로 이루어진 긴 context를 처리하는 task에서 더 좋은 성능 예상

## 실행 코드

### train_retrieval.py / train_reader.py

```bash
wandb sweep config_retrieval.yaml  ## retrieval 학습
wandb sweep config_reader.yaml  ## reader 학습
wandb agent SWEEP_ID --count 5 ## SWEEP_ID에 위에서 반환된 sweep id   ## --count 뒤에는 반복 실험 진행할 횟수
```

### context_sparse_embedding.py

```bash
# -m : model name (AutoModel.frompretrained()에 넣는 model name)
# -k, -b, -e : bm25 parameter (optional, float)
python context_sparse_embedding.py -m jhgan/ko-sroberta-multitask
```

### context_dense_embedding.py

```bash
# -mp : model path (artifact 상의 model path, 하단 첫 번째 이미지 빨간 밑줄)
# -mn : model name (artifact 상의 model name, 하단 두 번째 이미지 빨간 밑줄)
# -b : batch size (optional, int)
python3 context_dense_embedding.py -mp [model path] -mn [model name]
```

### test.py

```bash
# -rtmp : retrieval model path (artifact 상의 model path)
# -rtmn : retrieval model name (artifact 상의 model name)
# -rdmp : reader model path (artifact 상의 model path)
# -rdmn : reader model name (artifact 상의 model name)
# -k : number of selected contexts (optional, int)
# -w : weight for dense embedding in hybrid model (optional, float, 0~1)
python3 test.py -rtmp [retrieval model path] -rtmn [retrieval model name] -rdmp [reader model path] -rdmn [reader model name]
```

### inference.py

```bash
python3 inference.py -rtmp [retrieval model path] -rtmn [retrieval model name] -rdmp [reader model path] -rdmn [reader model name]
```
