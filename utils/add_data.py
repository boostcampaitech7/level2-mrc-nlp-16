import json

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

# 데이터셋 로드
dataset = load_dataset("squad_kor_v1")

# train 데이터를 DataFrame으로 변환
korquad_train = pd.DataFrame(dataset["train"])

# validation 데이터를 DataFrame으로 변환
korquad_valid = pd.DataFrame(dataset["validation"])

# answers 컬럼 처리 (리스트 형태의 첫 번째 답변만 추출)
korquad_train["answer"] = korquad_train["answers"].apply(lambda x: x["text"][0])
korquad_valid["answer"] = korquad_valid["answers"].apply(lambda x: x["text"][0])

# 원본 answers 컬럼 제거
korquad_train.drop(columns=["answers", "id", "title"], inplace=True)
korquad_valid.drop(columns=["answers", "id", "title"], inplace=True)

# # 결과 확인
# print("Training Data Shape:", korquad_train.shape)
# print("\nValidation Data Shape:", korquad_valid.shape)
# print("\nTraining Data Sample:")
# print(korquad_train.head())


def extract_qa(data):
    questions = []
    answers = []
    contexts = []
    # ids = []

    for paragraph in data:
        paragraph = paragraph["paragraphs"][0]
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            questions.append(qa["question"])
            answers.append(qa["answers"][0]["text"])  # 첫 번째 답변의 텍스트만 추출
            contexts.append(context)
            # ids.append(qa['id'])

    return {
        "question": questions,
        "answer": answers,
        "context": contexts,
        # 'id': ids
    }


# 데이터를 DataFrame으로 변환
base_path = "./data"
data = json.load(open(base_path + "/ko_wiki_v1_squad.json"))
qa_data = extract_qa(data["data"])
wiki_df = pd.DataFrame(qa_data)

# 데이터프레임 출력
# print(wiki_df)

##############################################################


# 1. contexts 데이터 확장 함수 수정
def extend_contexts(existing_contexts, new_df):
    # 마지막 document_id 찾기
    last_id = max([int(k) for k in existing_contexts.keys()])

    # 새로운 contexts 생성
    new_contexts = {}
    for idx, context in enumerate(new_df["context"].unique(), start=last_id + 1):
        new_contexts[str(idx)] = context

    # 기존 contexts와 병합
    merged_contexts = {**existing_contexts, **new_contexts}

    # context to id mapping 생성 (나중에 document_id 할당에 사용)
    context_to_id = {v: str(k) for k, v in new_contexts.items()}  # 문자열로 저장

    return merged_contexts, context_to_id


# 2. 데이터셋 확장을 위한 함수 수정
def create_dataset_entry(row, context_to_id):
    return {
        "title": None,
        "context": row["context"],
        "question": row["question"],
        "id": None,
        "answers": {"answer_start": [row["context"].find(row["answer"])], "text": [row["answer"]]},
        "document_id": int(context_to_id[row["context"]]),  # 문자열을 정수로 변환
        "__index_level_0__": None,
    }


# 3. 데이터 병합 및 변환
def merge_datasets(train_dataset, new_dfs, contexts):
    # contexts 확장
    all_new_df = pd.concat(new_dfs, ignore_index=True)
    merged_contexts, context_to_id = extend_contexts(contexts, all_new_df)

    # 기존 데이터를 리스트로 변환
    dataset_list = [dict(item) for item in train_dataset]

    # 새로운 데이터 추가
    for _, row in tqdm(all_new_df.iterrows(), desc="Converting new data"):
        dataset_list.append(create_dataset_entry(row, context_to_id))

    # Dataset 객체로 변환
    new_dataset = Dataset.from_list(dataset_list)

    return new_dataset, merged_contexts


# 메인 실행 코드
# 1. 기존 데이터 로드
dataset = load_from_disk("data/train_dataset")
train_dataset = dataset["train"]

# contexts 로드 부분 수정
with open("data/wikipedia_documents.json", "r", encoding="utf-8") as f:
    contexts = json.load(f)
contexts = {str(value["document_id"]): value["text"] for value in contexts.values()}  # document_id를 문자열로 저장

# 2. 새로운 데이터셋 병합
new_datasets = [korquad_train, korquad_valid, wiki_df]
merged_dataset, merged_contexts = merge_datasets(train_dataset, new_datasets, contexts)

merged_contexts = {str(k): v for k, v in merged_contexts.items()}
# 3. 결과 저장
# 새로운 contexts 저장
with open("data/wikipedia_documents_extended.json", "w", encoding="utf-8") as f:
    json_contexts = {
        str(k): {
            "document_id": int(k),  # document_id는 정수형으로 저장
            "text": v
        } for k, v in merged_contexts.items()
    }
    json.dump(json_contexts, f, ensure_ascii=False, indent=4)

# with open("data/wikipedia_documents_extended.json", "w", encoding="utf-8") as f:
#     json.dump(merged_contexts, f, ensure_ascii=False, indent=4)
# 새로운 데이터셋 저장
merged_dataset.save_to_disk("data/train_dataset_extended")

# 결과 확인
print(f"Original dataset size: {len(train_dataset)}")
print(f"Merged dataset size: {len(merged_dataset)}")
print(f"Original contexts size: {len(contexts)}")
print(f"Merged contexts size: {len(merged_contexts)}")
