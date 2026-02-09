import re
import pandas as pd
from typing import List, Dict

# --- Configuration ---
# These special tokens allow the model to learn structure and verticality
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>' 
EOS_TOKEN = '<EOS>' 
NL_TOKEN = '<NL>'   # Newline Token
IND_TOKEN = '<IND>' # 4-Space Indent Token
UNK_TOKEN = '<UNK>' # Unknown Token

class CustomTokenizer:
    """
    A custom tokenizer specialized for Python code structure.
    Learns newlines and indentation as individual tokens for 'Pretty Code' generation.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the tokenizer and builds the vocabulary from the provided DataFrame.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        self._initialize_special_tokens()
        self._build_vocab(df)

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens, including special tokens."""
        return len(self.token_to_id)

    def _initialize_special_tokens(self):
        """Sets the required special tokens at fixed indices."""
        self._add_token(PAD_TOKEN) # ID 0
        self._add_token(SOS_TOKEN) # ID 1
        self._add_token(EOS_TOKEN) # ID 2
        self._add_token(NL_TOKEN)  # ID 3
        self._add_token(IND_TOKEN) # ID 4
        self._add_token(UNK_TOKEN) # ID 5
    
    def _add_token(self, token: str):
        """Adds a new token to the vocabulary if it doesn't already exist."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text while preserving formatting (newlines and indentation).
        """
        if not isinstance(text, str):
            return []
            
        # 1. Preserve Vertical Structure (Newlines)
        text = text.replace('\n', f' {NL_TOKEN} ')
        
        # 2. Preserve Horizontal Structure (4-space indentation)
        # We replace them with the token so the model treats 'one indent' as one concept
        text = text.replace('    ', f' {IND_TOKEN} ')
            
        # 3. Add spaces around punctuation/symbols (careful not to break our special tags)
        # This regex adds space around symbols but ignores words inside < >
        text = re.sub(r'(?<!<)([.?,:;()\[\]{}!=+-/*&^|])(?!>)', r' \1 ', text)
        
        # 4. Collapse multiple spaces and strip
        # Keep tokens lowercase EXCEPT for our special tokens
        raw_tokens = text.split()
        tokens = [t if t.startswith('<') and t.endswith('>') else t.lower() for t in raw_tokens]
        
        return tokens

    def _build_vocab(self, df: pd.DataFrame):
        """
        Iterates over both columns to build a unified vocabulary.
        Uses 'Problem' for prompts and 'Python Code' for solutions.
        """
        all_text = pd.concat([
            df['Problem'].astype(str),     
            df['Python Code'].astype(str) 
        ])
        
        for text in all_text:
            tokens = self.tokenize(text)
            for token in tokens:
                self._add_token(token)

    def encode(self, tokens: List[str]) -> List[int]:
        """Converts a list of token strings into a list of integer IDs."""
        unk_id = self.token_to_id[UNK_TOKEN] 
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Converts a list of integer IDs back into a formatted multi-line string."""
        tokens = [self.id_to_token.get(idx, UNK_TOKEN) for idx in token_ids]
        
        # Filter out start/end/pad tokens from the output string
        clean_tokens = [t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
        
        # Join with a single space initially
        decoded_str = ' '.join(clean_tokens)
        
        # REVERSE FORMATTING: Convert tokens back to actual whitespace
        decoded_str = decoded_str.replace(IND_TOKEN, '    ')
        decoded_str = decoded_str.replace(NL_TOKEN, '\n')
        
        # Clean up spaces around common punctuation for a professional look
        decoded_str = decoded_str.replace(' ( ', '(').replace(' ) ', ')').replace(' : ', ': ')
        decoded_str = decoded_str.replace(' . ', '.')
        
        return decoded_str.strip()