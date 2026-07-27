import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        norm: str = "pre",  # "pre" | "post" | "none", the section 7.3 layer-norm ablations
        ffn_type: str = "swiglu",  # "swiglu" | "silu", the section 7.3 gating ablation
        device: torch.device = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.norm = norm
        self.ffn_type = ffn_type
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta,
                    device=device, norm=norm, ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        # Only the pre-norm architecture needs a final norm: post-norm blocks already end in one,
        # and the no-norm ablation removes them all. Identity (not deletion) keeps the module tree
        # stable for checkpoints and for ActivationMonitor's hook on the residual stream.
        self.ln_final = RMSNorm(d_model=d_model, device=device) if norm == "pre" else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(
        self, in_indices: Int[Tensor, "batch_size sequence_length"]
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
