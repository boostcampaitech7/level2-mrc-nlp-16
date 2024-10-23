from pororo import Pororo
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')

# 1. 다중 언어 번역 모델 로드
mt = Pororo(task="translation", lang="multi")

# 2. 역번역 증강 함수 정의 (Pororo 자체 토크나이저 사용)
def augment_question(question):
    # 한국어 -> 영어 번역
    translated_en = mt(question, src="ko", tgt="en")  # 한국어 -> 영어 번역
    # 영어 -> 한국어 역번역
    back_translated_ko = mt(translated_en, src="en", tgt="ko")  # 영어 -> 한국어 역번역
    return back_translated_ko

# 3. 데이터 로드 (train 데이터만 로드)
dataset = load_from_disk("train_dataset")
train_dataset = dataset["train"]

# 4. train 데이터셋에 역번역 증강 적용 함수 정의 (TQDM 추가)
def augment_dataset_translation(dataset, num_aug=1):
    augmented_data = []
    # TQDM으로 진행 상황을 확인
    for example in tqdm(dataset, desc="Augmenting Data", unit="question"):
        original_question = example['question']
        for _ in range(num_aug):
            augmented_question = augment_question(original_question)
            new_example = example.copy()
            new_example['question'] = augmented_question
            augmented_data.append(new_example)
    return augmented_data

# 5. train 데이터에 증강 적용
augmented_train_data = augment_dataset_translation(train_dataset)

# 6. 원본 데이터와 증강된 데이터를 결합
combined_train_data = list(train_dataset) + augmented_train_data

# 7. 최종 데이터셋을 생성
final_train_dataset = Dataset.from_list(combined_train_data)

# 8. 최종 train 데이터셋을 저장
final_train_dataset.save_to_disk("Pororo_augmented_train_dataset")