import torch
import torch.nn as nn
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
import re

logger = logging.getLogger(__name__)


class DualLAATConfig:
    """Configuration class for DualLAAT model."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        
        # Encoder configuration
        encoder_type: str = 'rnn',  # 'rnn' or 'cnn'
        
        # RNN configuration
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_type: str = 'lstm',  # 'lstm' or 'gru'
        
        # CNN configuration  
        kernel_size: int = 10,
        num_filters: int = 128,
        
        # Common configuration
        dropout: float = 0.1,
        num_mha_heads: int = 1,
        projection_dim: int = 128,
        max_note_length: int = 4000,
        max_code_length: int = 48,
        
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type.lower()
        
        # RNN params
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type.lower()
        
        # CNN params
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        # Common params
        self.dropout = dropout
        self.num_mha_heads = num_mha_heads
        self.projection_dim = projection_dim

        # Data params
        self.max_note_length = max_note_length
        self.max_code_length = max_code_length

        # Keep params only relevant to encoder_type
        if self.encoder_type == 'rnn':
            # Remove CNN-specific params
            del self.kernel_size
            del self.num_filters
        elif self.encoder_type == 'cnn':
            # Remove RNN-specific params
            del self.rnn_hidden_size
            del self.rnn_num_layers
            del self.rnn_type
        
        # Sort attributes for consistency
        self.__dict__ = dict(sorted(self.__dict__.items()))
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)






class DualLAAT(nn.Module):
    """DualLAAT model with configurable encoder (RNN or CNN) and dual attention mechanism."""
    
    def __init__(
        self,
        config: Optional[DualLAATConfig] = None,
        word_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize config
        if config is None:
            config = DualLAATConfig(**kwargs)
        self.config = config
        
        # Validate encoder type
        if self.config.encoder_type not in ['rnn', 'cnn']:
            raise ValueError(f"encoder_type must be 'rnn' or 'cnn', got {self.config.encoder_type}")
        
        # Initialize embeddings
        if word_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(word_embeddings, dtype=torch.float32), 
                freeze=False
            )
            # Update config with actual embedding dimensions
            self.config.vocab_size = word_embeddings.shape[0]
            self.config.embedding_dim = word_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(
                self.config.vocab_size, 
                self.config.embedding_dim,
                padding_idx=0
            )
        
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        
        # Initialize encoder based on type
        if self.config.encoder_type == 'rnn':
            self._init_rnn_encoder()
        else:  # cnn
            self._init_cnn_encoder()

        # Initialize attention and final layers
        self._init_attention_layers()
        self._init_final_layer()
        
        # Initialize weights
        self._init_weights()

    def _init_rnn_encoder(self):
        """Initialize RNN encoder."""
        if self.config.rnn_type == 'lstm':
            self.encoder_text = nn.LSTM(
                input_size=self.config.embedding_dim,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                bidirectional=True,
                dropout=self.config.dropout if self.config.rnn_num_layers > 1 else 0,
                batch_first=True
            )
            self.encoder_code = nn.LSTM(
                input_size=self.config.embedding_dim,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                bidirectional=True,
                dropout=self.config.dropout if self.config.rnn_num_layers > 1 else 0,
                batch_first=True
            )
        elif self.config.rnn_type == 'gru':
            self.encoder_text = nn.GRU(
                input_size=self.config.embedding_dim,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                bidirectional=True,
                dropout=self.config.dropout if self.config.rnn_num_layers > 1 else 0,
                batch_first=True
            )
            self.encoder_code = nn.GRU(
                input_size=self.config.embedding_dim,
                hidden_size=self.config.rnn_hidden_size,
                num_layers=self.config.rnn_num_layers,
                bidirectional=True,
                dropout=self.config.dropout if self.config.rnn_num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.config.rnn_type}. Use 'lstm' or 'gru'.")
        
        self.text_dim = self.config.rnn_hidden_size * 2
        self.code_dim = self.config.rnn_hidden_size * 2

    def _init_cnn_encoder(self):
        """Initialize CNN encoder."""
        self.encoder_text = nn.Conv1d(
            in_channels=self.config.embedding_dim,
            out_channels=self.config.num_filters,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2
        )
        self.encoder_code = nn.Conv1d(
            in_channels=self.config.embedding_dim,
            out_channels=self.config.num_filters,
            kernel_size=self.config.kernel_size,
            padding=self.config.kernel_size // 2
        )
        
        self.text_dim = self.config.num_filters
        self.code_dim = self.config.num_filters

    def _init_attention_layers(self):
        """Initialize attention projection layers."""
        self.encode_dim = self.text_dim * self.config.num_mha_heads
        
        self.projection_text_layers = nn.ModuleList([
            nn.Linear(self.text_dim, self.config.projection_dim) 
            for _ in range(self.config.num_mha_heads)
        ])
        self.projection_code_layers = nn.ModuleList([
            nn.Linear(self.code_dim, self.config.projection_dim) 
            for _ in range(self.config.num_mha_heads)
        ])

    def _init_final_layer(self):
        """Initialize final classification layer."""
        self.final_layer = nn.Sequential(
            nn.Linear(self.encode_dim, self.encode_dim),
            nn.ReLU(),
            nn.Linear(self.encode_dim, 1)
        )

    def _init_weights(self):
        """Initialize model weights."""
        layers_to_init = []
        
        # Add encoder layers
        layers_to_init.extend([self.encoder_text, self.encoder_code])
        
        # Add attention layers
        layers_to_init.extend(self.projection_text_layers)
        layers_to_init.extend(self.projection_code_layers)
        
        # Add final layer
        layers_to_init.extend(self.final_layer)
        
        for layer in layers_to_init:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    
    @staticmethod
    def perform_attention(x_text, x_code, projection_text, projection_code, **kwargs):
        # Perform attention
        representation_text = torch.tanh(projection_text(x_text))
        representation_code = torch.tanh(projection_code(x_code))

        attention_weight = torch.softmax(
            representation_code.matmul(representation_text.transpose(1, 2)), dim=2
        )
        weighted_text = attention_weight @ x_text
        return weighted_text
    

    def _encode_text(self, x_text):
        """Encode text input."""
        if self.config.encoder_type == 'rnn':
            x_text, _ = self.encoder_text(x_text)
        else:  # cnn
            x_text = self.encoder_text(x_text.transpose(1, 2)).transpose(1, 2)
        return x_text
    
    def _encode_code(self, x_code):
        """Encode code input."""
        if self.config.encoder_type == 'rnn':
            # Get code query representation from final hidden state
            _, x_out = self.encoder_code(x_code)
            if isinstance(x_out, tuple):  # LSTM returns (h_n, c_n)
                x_out = x_out[0]  # Take hidden state
            x_code_query = x_out.transpose(0, 1).flatten(1, 2) 
        else:  # cnn
            x_code = self.encoder_code(x_code.transpose(1, 2)).transpose(1, 2)
            x_code_query = x_code.max(dim=1)[0]
        return x_code_query


    def forward(self, input_text, input_code, **kwargs):
        """Forward pass of the model."""
        # Encode text
        x_text = self.embedding(input_text)
        x_text = self.embedding_dropout(x_text)
        x_text = self._encode_text(x_text)
        
        # Encode code
        x_code = self.embedding(input_code)
        x_code_query = self._encode_code(x_code)

        # Multi-head attention
        mha_outputs = [
            self.perform_attention(x_text, x_code_query, projection_text, projection_code)
            for projection_text, projection_code in zip(
                self.projection_text_layers, self.projection_code_layers
            )
        ]
        mha_outputs = torch.cat(mha_outputs, dim=2)
        
        # Final classification
        logits = self.final_layer(mha_outputs).squeeze(2)
        return logits



    def _tokenize_text(self, text: str, max_length: int, token2id: Optional[Dict[str, int]]) -> List[int]:
        """Simple tokenization function."""
        if token2id is None:
            assert self.token2id is not None, "token2id dictionary must be provided."
        else:
            self.token2id = token2id
        ids = [self.token2id[t] if t in self.token2id else self.token2id['<UNK>'] for t in text.split()]

        padded = np.full((max_length), self.token2id['<PAD>'], dtype=int)
        length = min(len(ids), max_length)
        padded[:length] = ids[:length]
        return padded
    

    def tokenize(
        self, 
        note_inputs: Optional[Union[str, List[str]]] = None, 
        code_inputs: Optional[Union[str, List[str]]] = None,
        preprocess: bool = True,
        token2id: Optional[Dict[str, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize note and code inputs."""

        assert note_inputs is not None or code_inputs is not None, "At least one of note_inputs or code_inputs must be provided."

        # Handle single string inputs
        if isinstance(note_inputs, str):
            note_inputs = [note_inputs]
        if isinstance(code_inputs, str):
            code_inputs = [code_inputs]

        assert self.token2id is not None or token2id is not None, "token2id dictionary must be provided."
        if token2id is not None:
            self.token2id = token2id

        # Preprocess note inputs
        if preprocess and note_inputs is not None:
            text_preprocessor = TextPreprocessor()
            note_inputs = [text_preprocessor(text) for text in note_inputs]

        # Preprocess code inputs
        if preprocess and code_inputs is not None:
            code_inputs = [code.lower() for code in code_inputs]
            max_desc_length = max([len(desc.split()) for desc in code_inputs])
            if max_desc_length > self.config.max_code_length:
                logger.warning(f"Some code descriptions exceed max_code_length of {self.config.max_code_length}. They will be truncated.")
                max_desc_length = self.config.max_code_length

        # Tokenize inputs
        output = {}
        if note_inputs is not None:
            note_token_ids = [self._tokenize_text(text, self.config.max_note_length, self.token2id) for text in note_inputs]
            output['input_text'] = torch.tensor(np.array(note_token_ids), dtype=torch.long)
        if code_inputs is not None:
            code_token_ids = [self._tokenize_text(code, max_desc_length, self.token2id) for code in code_inputs]
            output['input_code'] = torch.tensor(np.array(code_token_ids), dtype=torch.long)

        return output


    def predict(
        self, 
        notes_to_code: Union[str, List[str]], 
        codes_to_consider: Union[str, List[str]],
        token2id: Optional[Dict[str, int]] = None,
        preprocess: bool = True,
        batch_size: int = 32,
    ) -> Union[torch.Tensor, tuple]:
        """Predict similarity scores for text-code pairs."""
        self.eval()

        # Move model to cuda if available
        if torch.cuda.is_available():
            self.to('cuda')
            logger.info("Model moved to GPU for prediction.")

        device = next(self.parameters()).device


        # Encode code inputs
        input_code = self.tokenize(note_inputs=None, code_inputs=codes_to_consider, preprocess=preprocess, token2id=token2id)['input_code']

        # Tokenize notes 
        if isinstance(notes_to_code, str):
            note_inputs = notes_to_code
        else:
            note_inputs = notes_to_code

        input_text = self.tokenize(note_inputs=note_inputs, code_inputs=None, preprocess=preprocess, token2id=token2id)['input_text']


        all_logits = []
        num_batches = (len(notes_to_code) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(notes_to_code), batch_size), desc="Inferencing on notes", total=num_batches):
            batch_inputs = {
                "input_text": input_text[i:i+batch_size],
                "input_code": input_code
            }

            with torch.no_grad():
                # Move to same device as model
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

                logits = self.forward(**batch_inputs)
                all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        return {
            'logits': all_logits,
            'probabilities': torch.sigmoid(all_logits)
        }

        
       
    def save_pretrained(
        self, 
        save_directory: str, 
        token2id: Optional[Dict[str, int]] = None,
        save_function: Optional[callable] = None
    ):
        """Save model checkpoint and configuration."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model state dict
        model_path = save_directory / "pytorch_model.bin"
        if save_function is not None:
            save_function(self.state_dict(), model_path)
        else:
            torch.save(self.state_dict(), model_path)
        
        # Save tokenizer if provided
        if token2id is not None:
            tokenizer_path = save_directory / "token2id.json"
            with open(tokenizer_path, 'w') as f:
                json.dump(token2id, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        map_location: Optional[str] = None,
        **kwargs
    ):
        """Load model from checkpoint."""
        model_path = Path(model_path)
        
        # Load config
        config = DualLAATConfig.from_pretrained(model_path)
        
        # Create model instance
        model = cls(config=config, **kwargs)
        
        # Load state dict
        state_dict_path = model_path / "pytorch_model.bin"
        if state_dict_path.exists():
            state_dict = torch.load(
                state_dict_path, 
                map_location=map_location or 'cpu'
            )
            model.load_state_dict(state_dict)
        else:
            logger.warning(f"No model weights found at {state_dict_path}")

        # Load tokenizer if available
        token2id = cls.load_token2id(model_path)
        if token2id is not None:
            model.token2id = token2id
        
        return model

    @staticmethod
    def load_token2id(model_path: str) -> Optional[Dict[str, int]]:
        """Load token2id dictionary from model directory."""
        tokenizer_path = Path(model_path) / "token2id.json"
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r') as f:
                return json.load(f)
        return None


    def num_parameters(self, only_trainable: bool = True):
        """Get number of parameters in the model."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())




class TextPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = True,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
    ) -> None:
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters

    def __call__(self, text: str) -> str:
        
        if self.lower:
            text = text.lower()
        if self.convert_danish_characters:
            text = re.sub(r"å", "aa", text)
            text = re.sub(r"æ", "ae", text)
            text = re.sub(r"ø", "oe", text)
        if self.remove_accents:
            text = re.sub(r"é|è|ê", "e", text)
            text = re.sub(r"á|à|â", "a", text)
            text = re.sub(r"ô|ó|ò", "o", text)
        if self.remove_brackets:
            text = re.sub(r"\[[^]]*\]", "", text)
        if self.remove_special_characters:
            text = re.sub(r"\n|/|-", " ", text)
            text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        if self.remove_special_characters_mullenbach:
            text = re.sub(r"[^A-Za-z0-9]+", " ", text)
        if self.remove_digits:
            text = re.sub(r"(\s\d+)+\s", " ", text)

        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text
