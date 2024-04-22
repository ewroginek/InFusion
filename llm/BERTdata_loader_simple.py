import torch
from torch.utils.data import Dataset, DataLoader

class BERTDataset(Dataset):
    def __init__(self, text, labels, max_length, tokenizer,  val=False):
        self.data = text
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.flag = val
   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        # Tokenize the text
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                              return_tensors='pt')

        input_ids = encodings["input_ids"].squeeze(0)
        masks = encodings["attention_mask"].squeeze(0)
        # ttids = encodings["token_type_ids"].squeeze(0)

        # Convert attention_score to a PyTorch tensor
        # attention_score = torch.tensor(attention_score, dtype=torch.float32)

        # Pad or truncate attention_score_padded
        num_tokens = input_ids.size(0)
        # attention_score_padded = torch.cat([attention_score[:num_tokens],
        #                                     torch.zeros(max(0, num_tokens - len(attention_score)),
        #                                                 dtype=torch.float32)])

        # embedding_index = torch.arange(self.projection_dim).squeeze(0)

        return {"input_ids": input_ids, "attention_masks": masks,
                 "label": label}
  

    @classmethod
    def dataprep(cls, train_df, val_df, tokenizer, MAX_LEN, BATCH_SIZE):
        val_ds = cls(val_df["tweet"].values, val_df["class"].values,
                     max_length=MAX_LEN, tokenizer=tokenizer)
        train_ds = cls(train_df["tweet"].values, train_df["class"].values, 
                       max_length=MAX_LEN, tokenizer=tokenizer)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        return train_loader, val_loader
