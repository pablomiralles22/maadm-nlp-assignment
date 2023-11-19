import os
import random
import re

from torch.utils.data import Dataset

class BlogDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(path))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        author_path = os.path.join(self.path, str(index))
        author_files = os.listdir(author_path)

        post1_ind, post2_ind = random.sample(author_files, 2)
        post1_path = os.path.join(author_path, post1_ind)
        post2_path = os.path.join(author_path, post2_ind)

        with open(post1_path, "r", encoding="utf-8") as f:
            post1 = f.read()
        with open(post2_path, "r", encoding="utf-8") as f:
            post2 = f.read()

        return {
            "post1": post1,
            "post2": post2,
        }

class BlogCollatorFn:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        posts = []
        for item in batch:
            posts.append(self.__clean_text(item["post1"]))
            posts.append(self.__clean_text(item["post2"]))

        encoding = self.tokenizer.batch_encode_plus(
            posts,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    def __clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        if len(words) > self.max_len:
            start = random.randint(0, len(words) - self.max_len)
            words = words[start:start + self.max_len]
        return " ".join(words)
