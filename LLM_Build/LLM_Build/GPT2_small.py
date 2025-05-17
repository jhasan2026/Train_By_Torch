import torch
import torch.nn as nn

# %%
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "num_of_heads": 12,
    "num_of_layers": 12,
    "drop_rate_attention": 0.1,
    "drop_rate_embedding": 0.1,
    "drop_rate_shortcut": 0.1,
    "qkv_bias": False
}


# %%

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_of_head, qkv_bias=False):
        super().__init__()

        assert (d_out % num_of_head == 0), "d_out must be divisible by num_of_head"

        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.num_of_head = num_of_head

        self.head_dim = d_out // num_of_head
        self.out_proj = nn.Linear(d_out, d_out)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        num_of_inputs, num_of_tokens, embedding_dim = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(num_of_inputs, num_of_tokens, self.num_of_head, self.head_dim)
        keys = keys.view(num_of_inputs, num_of_tokens, self.num_of_head, self.head_dim)
        values = values.view(num_of_inputs, num_of_tokens, self.num_of_head, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_of_tokens, :num_of_tokens]

        attention_score.masked_fill_(mask_bool, -torch.inf)

        k_d = keys.shape[-1]
        attention_weight = torch.softmax(attention_score / k_d ** 0.5, dim=-1)
        attention_weight = self.dropout(attention_weight)

        context_vector = (attention_weight @ values).transpose(1, 2)

        context_vector = context_vector.contiguous().view(
            num_of_inputs, num_of_tokens, self.d_out
        )
        context_vector = self.out_proj(context_vector)
        return context_vector


class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x_normalized + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
                1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.44715 * torch.pow(x, 3)))
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embedding_dim'], 4 * config['embedding_dim']),
            GELU(),
            nn.Linear(4 * config['embedding_dim'], config['embedding_dim'])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = LayerNormalization(config['embedding_dim'])
        self.attention = MultiHeadAttention(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            context_length=config['context_length'],
            dropout=config['drop_rate_attention'],
            num_of_head=config['num_of_heads'],
            qkv_bias=config['qkv_bias']
        )
        self.dropout_shortcut = nn.Dropout(config['drop_rate_shortcut'])
        self.layer_norm2 = LayerNormalization(config['embedding_dim'])
        self.ff = FeedForward(config)

    def forward(self, x):
        shortcut = x

        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout_shortcut(x)

        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)

        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.positional_embedding = nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout_embedding = nn.Dropout(config['drop_rate_embedding'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['num_of_layers'])]
        )

        self.final_normalization = LayerNormalization(config['embedding_dim'])

        self.output_head = nn.Linear(config['embedding_dim'], config['vocab_size'], bias=False)

    def forward(self, inputs):
        batch_size, no_of_token = inputs.shape
        token_embedding = self.token_embedding(inputs)
        positional_embedding = self.positional_embedding(
            torch.arange(no_of_token, device=inputs.device)
        )

        x = token_embedding + positional_embedding
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_normalization(x)
        logits = self.output_head(x)
        return logits



def generate_text(model, inputs, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        inputs_conditional = inputs[:, -context_size:]
        with torch.no_grad():
            logits = model(inputs_conditional)

        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_word_index = torch.argmax(probabilities, dim=-1, keepdim=True)
        inputs = torch.concat((inputs, next_word_index), dim=1)
    return inputs