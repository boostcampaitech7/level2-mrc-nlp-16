from pororo import Pororo
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')

# 역번역 증강 함수 정의 (Pororo 자체 토크나이저 사용)
def back_trans_augmentation(question, mt):
    # 한국어 -> 영어 번역
    translated_en = mt(question, src="ko", tgt="en")  # 한국어 -> 영어 번역
    # 영어 -> 한국어 역번역
    back_translated_ko = mt(translated_en, src="en", tgt="ko")  # 영어 -> 한국어 역번역
    return back_translated_ko

# train 데이터셋에 역번역 증강 적용 함수 정의 (TQDM 추가)
def apply_back_trans_augmentation(dataset, mt, num_aug=1):
    augmented_data = []
    for example in tqdm(dataset, desc="Augmenting Data", unit="question"):
        original_question = example['question']
        for _ in range(num_aug):
            augmented_question = back_trans_augmentation(original_question, mt)
            new_example = example.copy()
            new_example['question'] = augmented_question
            augmented_data.append(new_example)
    return augmented_data

def main():
    # 다중 언어 번역 모델 로드
    mt = Pororo(task="translation", lang="multi")

    # 데이터 로드 (train 데이터만 로드)
    train_dataset_path = "../data/train_dataset"
    dataset = load_from_disk(train_dataset_path)
    train_dataset = dataset["train"]
    
    # 데이터 증강
    augmented_train_data = apply_back_trans_augmentation(train_dataset, mt) # train 데이터에 증강 적용
    combined_train_data = list(train_dataset) + augmented_train_data # 원본 데이터와 증강된 데이터를 결합
    final_train_dataset = Dataset.from_list(combined_train_data) # 최종 데이터셋을 생성

    # 데이터 저장
    final_train_dataset.save_to_disk("Pororo_augmented_train_dataset") # 최종 train 데이터셋을 저장

if __name__ == "__main__":
    main()