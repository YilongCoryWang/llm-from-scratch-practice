import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
from importlib.metadata import version
import tiktoken

# print("tokenizer version", version("tiktoken"))

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
# print(raw_text[:99])

# tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # text => index
        # print(len(token_ids), sorted(token_ids))
        for i in range(0, len(token_ids) - max_length, stride):
            # sliding window
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+ 1:i + max_length + 1] # 右移一个
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt,
            batch_size=4, # 每次从dataset即input_chunk, output_chunk对中取出几对数据。值小省内存，但会增加噪声
            max_length=256, # size of sliding window, e.g. how many words does it process at a time
            stride=128,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
# Token IDs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
print("\nInputs.shape:\n", inputs.shape) # torch.Size([8, 4]) # batch_size是8，所以有8个文本，每文本有 4 token

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) # torch.Size([8, 4, 256]) # each ID is mapped to a 256 dim vector

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("pos_embeddings.shape", pos_embeddings.shape) # torch.Size([4, 256])

input_embedding = token_embeddings + pos_embeddings
print("input_embedding.shape", input_embedding.shape) # torch.Size([8, 4, 256])