import json
import copy

class GPTConfig(object):
    """Configuration for `GPTModel`."""

    def __init__(self,
                 vocab_size,
                 n_positions=1024,
                 n_ctx=1024,
                 n_embd=768,
                 n_layer=12,
                 n_head=12,
                 layer_norm_epsilon=1e-5,
                 initializer_range=0.02,
                 dropout=0.1):
        """Constructs GPTConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `GPTModel`.
            n_positions: Number of positional embeddings.
            n_ctx: Context size (window size).
            n_embd: Size of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer decoder.
            n_head: Number of attention heads for each attention layer.
            layer_norm_epsilon: Epsilon for layer normalization.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
            dropout: The dropout probability.
        """
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.dropout = dropout

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `GPTConfig` from a Python dictionary of parameters."""
        config = GPTConfig(vocab_size=None)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `GPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
