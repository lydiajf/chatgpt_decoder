from pathlib import Path
import sentencepiece as spm

in_dir = Path(__file__).parent.parent / "final_model/all_text.txt"

spm.SentencePieceTrainer.train(
    input=str(in_dir),  # Input file
    model_prefix="tokenizer",  # Output model prefix
    vocab_size=16458,  # Vocabulary size
    model_type="unigram",  # Model type ('unigram', 'bpe', 'char', 'word')
    character_coverage=0.9995,  # Amount of characters covered by the model
)