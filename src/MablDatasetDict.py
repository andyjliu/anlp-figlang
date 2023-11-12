from datasets import Dataset, DatasetDict
import os


class MablDatasetDict(DatasetDict):

    def __init__(self, data_dir='data'):
        super().__init__()

        self.dataset_dict = {}
        for split in ['train', 'dev']:
            file_path = f"{data_dir}/{split}/en.csv"
            self.dataset_dict[split] = Dataset.from_csv(file_path)

        test_dir = f"{data_dir}/test"
        for file_name in os.listdir(test_dir):
            lang_short_name = file_name.split("_")[0]
            file_path = f"{test_dir}/{file_name}"
            self.dataset_dict[f"test-{lang_short_name}"] = Dataset.from_csv(file_path)

    def get_dataset_dict(self):
        return self.dataset_dict


if __name__ == "__main__":
    print(MablDatasetDict(data_dir='../data').get_dataset_dict())