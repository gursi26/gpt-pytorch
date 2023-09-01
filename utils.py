import torch
from transformers import OpenAIGPTTokenizer
from model import GPT, GPTSemanticSimilarity


def init_finetuning_model_and_tokenizer(weights_path, device):
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    gpt = GPT(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        d_model=768,
        num_heads=12,
        n_layers=12,
        p=0.1
    ).to(device)
    gpt.load_state_dict(torch.load(weights_path, map_location=device))
    
    special_tokens_dict = {"bos_token":"<bos>", "eos_token":"<eos>", "sep_token":"<sep>", "pad_token":"<pad>"}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    new_embedding_weights = torch.randn(num_added, 768).to(device)
    gpt.tokens_embed.weight.data = torch.cat([gpt.tokens_embed.weight.data, new_embedding_weights], dim=0)
    gpt = GPTSemanticSimilarity(gpt).to(device)
    return tokenizer, gpt


def inference(q1, q2, model, tokenizer, device):
    order1 = f"<bos> {q1} <sep> {q2} <eos>"
    order2 = f"<bos> {q2} <sep> {q1} <eos>"

    tokenized_input, mask = tokenizer([order1, order2], return_tensors="pt").values()
    tokenized_input, mask = tokenized_input.unsqueeze(1).to(device), mask.unsqueeze(1).to(device).type(torch.bool)

    output = model(tokenized_input[0], tokenized_input[1], mask[0], mask[1])
    return output.sigmoid().item()