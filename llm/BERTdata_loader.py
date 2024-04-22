import torch
from torch.utils.data import Dataset, DataLoader

class BERTDataset_Attention(Dataset):
    def __init__(self, text, labels, max_length, tokenizer, projection_dim,attention_scores, val=False):
        self.data = text
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.flag = val
        self.projection_dim = projection_dim
        self.attention_scores = attention_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        attention_score = self.attention_scores[idx]

        # Tokenize the text
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                              return_tensors='pt')

        input_ids = encodings["input_ids"].squeeze(0)
        masks = encodings["attention_mask"].squeeze(0)
        ttids = encodings["token_type_ids"].squeeze(0)

        # Convert attention_score to a PyTorch tensor
        attention_score = torch.tensor(attention_score, dtype=torch.float32)

        # Pad or truncate attention_score_padded
        num_tokens = input_ids.size(0)
        attention_score_padded = torch.cat([attention_score[:num_tokens],
                                            torch.zeros(max(0, num_tokens - len(attention_score)),
                                                        dtype=torch.float32)])

        embedding_index = torch.arange(self.projection_dim).squeeze(0)

        return {"input_ids": input_ids, "attention_masks": masks, "space": embedding_index,
                "attention_score": attention_score_padded, "label": label}

    @classmethod
    def dataprep(cls, train_df, val_df, tokenizer, MAX_LEN, BATCH_SIZE, PROJECTION_DIM):
        # print(MAX_LEN)
        val_ds = cls(val_df["tweet"].values, val_df["class"].values, val_df["softmax_numbers"].values,
                     MAX_LEN, tokenizer, PROJECTION_DIM)
        
        train_ds = cls(train_df["tweet"].values, train_df["class"].values, train_df["softmax_numbers"].values,
                       MAX_LEN, tokenizer, PROJECTION_DIM)

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,shuffle=True)

        return train_loader, val_loader
    
    @classmethod
    def dataprep_lime(cls, test_df, tokenizer, MAX_LEN, BATCH_SIZE, PROJECTION_DIM):
        # print(MAX_LEN)
        dataprep_lime_ds = cls(test_df["tweet"].values, test_df["class"].values, test_df["softmax_numbers"].values,
                     max_length=MAX_LEN, tokenizer=tokenizer, projection_dim=PROJECTION_DIM)
 

        lime_loader = DataLoader(dataprep_lime, batch_size=BATCH_SIZE)

        return lime_loader

