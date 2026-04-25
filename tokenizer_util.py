import re
import pandas as pd
from typing import List, Dict

# --- Special Tokens for SQL ---


PAD_TOKEN   = '<PAD>'
SOS_TOKEN   = '<SOS>'
EOS_TOKEN   = '<EOS>'
UNK_TOKEN   = '<UNK>'
TABLE_TOKEN = '<TBL>' 
COLUMN_TOKEN= '<COL>'  
SCHEMA_SEP  = '<SEP>' 



class CustomTokenizer:
    """
    Tokenizer for NL2SQL.
    Preserves underscores in identifiers and handles SQL operators.
    """
    def __init__(self, df: pd.DataFrame = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._initialize_special_tokens()
        if df is not None:
            self._build_vocab(df)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _initialize_special_tokens(self):
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, 
                      TABLE_TOKEN, COLUMN_TOKEN, SCHEMA_SEP]:
            self._add_token(token)

    def _add_token(self, token: str):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []

        # 1. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # 2. Add spaces around SQL punctuation/operators (except underscores)
        # This keeps "user_id" as one unit but separates "id=1" into "id = 1"
        text = re.sub(r'([,.;()=<>!+*/-])', r' \1 ', text)

        # 3. Clean up multi-char operators
        text = text.replace(' !  = ', ' != ').replace(' >  = ', ' >= ').replace(' <  = ', ' <= ')

        # 4. Final split and lowercase (except special tokens)
        raw_tokens = text.split()
        tokens = [t if t.startswith('<') and t.endswith('>') else t.lower()
                  for t in raw_tokens]

        return tokens

    def _build_vocab(self, df: pd.DataFrame):
        # Assuming columns 'Question' and 'SQL'
        all_text = pd.concat([
            df['Question'].astype(str),
            df['SQL'].astype(str)
        ])
        for text in all_text:
            for token in self.tokenize(text):
                self._add_token(token)
        print(f"SQL Vocab built: {self.vocab_size} tokens")

    def encode(self, tokens: List[str]) -> List[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.id_to_token.get(idx, UNK_TOKEN) for idx in token_ids]
        clean_tokens = [t for t in tokens if t not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
        
        # SQL Formatting
        decoded_str = ' '.join(clean_tokens)
        decoded_str = decoded_str.replace(' ( ', '(').replace(' ) ', ')')
        decoded_str = decoded_str.replace(' , ', ', ')
        decoded_str = decoded_str.replace(' . ', '.')
        
        return decoded_str.strip()