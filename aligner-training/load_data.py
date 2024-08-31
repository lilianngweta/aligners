import datasets
from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, SENTENCE_END


def load_aligner_dataset(dataset_path, seed=42):
    dataset = datasets.load_dataset("csv",  split='train', data_files=dataset_path)
    dataset = datasets.DatasetDict({"train":dataset})

    updated_dataset = dataset.map(
        lambda example: {
            "text": str(example["input"]) + INSTRUCTION_SEPARATOR + str(example["initial_response"]) + SENTENCE_SEPARATOR + str(example["aligned_response"]) + SENTENCE_END
        }
    )
    return updated_dataset.shuffle(seed=seed)


def load(dataset_path):
    ds = load_aligner_dataset(dataset_path)

    # Split into 70% train, 15% validation, 15% test
    ds = ds["train"]
    ds_train_testvalid = ds.train_test_split(
        test_size=0.3
    )  
    ds_test_valid = ds_train_testvalid["test"].train_test_split(test_size=0.5)

    ds_final = datasets.DatasetDict(
        {
            "train": ds_train_testvalid["train"],
            "validation": ds_test_valid["train"],
            "test": ds_test_valid["test"],
        }
    )
    return ds_final