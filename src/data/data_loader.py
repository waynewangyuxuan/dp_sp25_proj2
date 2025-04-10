from datasets import load_dataset
from transformers import RobertaTokenizer
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset

class AGNewsDataLoader:
    def __init__(self, model_name: str = 'roberta-base', max_length: int = 512):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def load_and_preprocess(self) -> Tuple[Dataset, Dataset]:
        """
        Load and preprocess the AG News dataset.
        
        Returns:
            Tuple[Dataset, Dataset]: Training and evaluation datasets
        """
        # Load dataset
        dataset = load_dataset('ag_news', split='train')
        
        # Preprocess function
        def preprocess(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        # Preprocess dataset
        tokenized_dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=['text']
        )
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
        # Split dataset
        split_datasets = tokenized_dataset.train_test_split(
            test_size=640,
            seed=42
        )
        
        return split_datasets['train'], split_datasets['test'] 