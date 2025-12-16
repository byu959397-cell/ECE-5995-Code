# trace_dataset.py
import os
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TRACETaskInfo:
    task_id: str
    task_name: str
    task_type: str
    description: str
    metrics: List[str]


TRACE_TASKS = {
    "c_stance": TRACETaskInfo("c_stance", "C-STANCE", "classification", "Chinese stance detection", ["accuracy"]),
    "fomc": TRACETaskInfo("fomc", "FOMC", "classification", "Federal Reserve policy classification", ["accuracy"]),
    # "meeting_bank": TRACETaskInfo("meeting_bank", "MeetingBank", "generation", "Meeting summarization", ["rougeL"]),
    # "py150": TRACETaskInfo("py150", "Py150", "generation", "Python code completion", ["rougeL"]),
    # "scienceqa": TRACETaskInfo("scienceqa", "ScienceQA", "classification", "Science question answering", ["accuracy"]),
    "numglue_cm": TRACETaskInfo("numglue_cm", "NumGLUE-cm", "generation", "Math commonsense reasoning", ["accuracy"]),
    "numglue_ds": TRACETaskInfo("numglue_ds", "NumGLUE-ds", "generation", "Math data science reasoning", ["accuracy"]),
    # "20minuten": TRACETaskInfo("20minuten", "20Minuten", "generation", "German news summarization", ["rougeL"]),
}

TASK_ID_TO_FOLDER = {
    "c_stance": "C-STANCE", "fomc": "FOMC", "meeting_bank": "MeetingBank",
    "py150": "Py150", "scienceqa": "ScienceQA", "numglue_cm": "NumGLUE-cm",
    "numglue_ds": "NumGLUE-ds", "20minuten": "20Minuten",
}


class TRACEDataset(Dataset):
    def __init__(
        self, 
        task_id: str, 
        split: str, 
        tokenizer, 
        max_seq_length: int = 1024,
        data_path: str = None, 
        max_samples: int = None, 
        for_training: bool = True,
        min_answer_tokens: int = 100
    ):
        self.task_id = task_id
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_info = TRACE_TASKS.get(task_id)
        self.for_training = for_training
        self.min_answer_tokens = min_answer_tokens
        
        self.data = self._load_data(data_path, max_samples)
        logger.info(f"Loaded {len(self.data)} samples for {task_id}/{split}")

    
    def _load_data(self, data_path: str, max_samples: int) -> List[Dict]:
        folder_name = TASK_ID_TO_FOLDER.get(self.task_id)
        file_path = os.path.join(data_path, folder_name, f"{self.split}.json")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        original_len = len(data)
        data = [d for d in data if d.get('answer', '').strip() and d.get('prompt', '').strip()]
        
        if len(data) < original_len:
            logger.warning(
                f"Filtered {original_len - len(data)} samples with empty answers"
                f"({len(data)}/{original_len} remaining)"
            )
        
        if max_samples and len(data) > max_samples:
            # 
            rng = random.Random(42) 
            data = rng.sample(data, max_samples)
            logger.info(f"Sampled {max_samples} from {len(data)} samples")
        
        return data
    
    def __len__(self):
        return len(self.data)

    # according to index return data samples
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt')
        answer = item.get('answer')
        
        if self.for_training:
            return self._tokenize_for_training(prompt, answer)
        else:
            return self._tokenize_for_eval(prompt, answer)
    
    def _tokenize_for_training(self, prompt: str, answer: str) -> Dict[str, Any]:
        if not answer or not answer.strip():
            logger.debug(f"Empty answer detected, creating empty sample")
            return self._create_empty_sample(prompt, answer)
      
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(
            answer + self.tokenizer.eos_token, 
            add_special_tokens=False
        )
        
        # truncate 
        answer_len = len(answer_ids)
        # answer length: 100-256 
        reserved_for_answer =  max(self.min_answer_tokens, min(answer_len, 256))
 
        max_prompt_len = self.max_seq_length - reserved_for_answer
        
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
            logger.debug(
                f"Truncated prompt from {len(prompt_ids)} to {max_prompt_len} tokens"
            )
        
        input_ids = prompt_ids + answer_ids
        
        # total length truncation
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
        
        # convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # right padding
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100 # mask prompt
        labels[attention_mask == 0] = -100 # mask padding
     
        valid_label_count = (labels != -100).sum().item()
        if valid_label_count == 0:
            logger.warning(
                f"No valid labels: prompt_len={len(prompt_ids)}, "
                f"answer_len={len(answer_ids)}, max_seq_len={self.max_seq_length}"
            )
            return self._create_empty_sample(prompt, answer)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_prompt': prompt,
            'raw_answer': answer,
            '_is_empty': False,
            '_prompt_len': len(prompt_ids),
            '_answer_len': len(answer_ids),
        }


    def _create_empty_sample(self, prompt: str = '', answer: str = '') -> Dict[str, Any]:
        input_ids = torch.full(
            (self.max_seq_length,), 
            self.tokenizer.pad_token_id, 
            dtype=torch.long
        )
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        labels = torch.full((self.max_seq_length,), -100, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'raw_prompt': prompt,
            'raw_answer': answer,
            '_is_empty': True,
        }
  
    # return each batch
    def _tokenize_for_eval(self, prompt: str, answer: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        enc = self.tokenizer(
            prompt_text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'raw_prompt': prompt,
            'raw_answer': answer,
            '_is_empty': False,
        }


def train_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    valid_batch = [x for x in batch if not x.get('_is_empty', False)]

    if len(valid_batch) == 0:
        logger.warning("Entire batch is empty, returning skip marker")
        return {'_skip_batch': True}
    
    if len(valid_batch) < len(batch):
        logger.debug(
            f"Filtered {len(batch) - len(valid_batch)}/{len(batch)} empty samples"
        )
    
    # Stack tensors
    result = {
        'input_ids': torch.stack([x['input_ids'] for x in valid_batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in valid_batch]),
        'labels': torch.stack([x['labels'] for x in valid_batch]),
    }
    

    if 'raw_prompt' in valid_batch[0]:
        result['raw_prompt'] = [x['raw_prompt'] for x in valid_batch]
        result['raw_answer'] = [x['raw_answer'] for x in valid_batch]

    if '_prompt_len' in valid_batch[0]:
        result['_stats'] = {
            'batch_size': len(valid_batch),
            'avg_prompt_len': sum(x['_prompt_len'] for x in valid_batch) / len(valid_batch),
            'avg_answer_len': sum(x['_answer_len'] for x in valid_batch) / len(valid_batch),
        }

    return result


# left padding
def eval_collate_fn(batch: List[Dict], tokenizer) -> Dict[str, Any]:
    batch = [x for x in batch if not x.get('_is_empty', False)]
    
    if len(batch) == 0:
        logger.warning("Entire eval batch is empty")
        return {'_skip_batch': True}

    max_len = max(x['input_ids'].shape[0] for x in batch)
    pad_id = tokenizer.pad_token_id
    
    padded_ids = []
    padded_mask = []
    
    for x in batch:
        seq_len = x['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            # left padding
            ids = torch.cat([
                torch.full((pad_len,), pad_id, dtype=x['input_ids'].dtype),
                x['input_ids']
            ])
            mask = torch.cat([
                torch.zeros(pad_len, dtype=x['attention_mask'].dtype),
                x['attention_mask']
            ])
        else:
            ids = x['input_ids']
            mask = x['attention_mask']
        
        padded_ids.append(ids)
        padded_mask.append(mask)
    
    result = {
        'input_ids': torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_mask),
    }
    
    if 'raw_prompt' in batch[0]:
        result['raw_prompt'] = [x['raw_prompt'] for x in batch]
        result['raw_answer'] = [x['raw_answer'] for x in batch]
    
    return result


class TRACEDataModule:
    def __init__(
        self,
        tokenizer,
        data_path: str = None,
        max_seq_length: int = 1024,
        batch_size: int = 2,
        num_workers: int = 0,
        train_samples: int = 500,
        eval_samples: int = 200,
        test_samples: int = 300,
        min_answer_tokens: int = 100,
    ):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.test_samples = test_samples
        self.min_answer_tokens = min_answer_tokens

        # 
        self._datasets: Dict[str, Dict[str, TRACEDataset]] = {}

    
    def get_task_ids(self) -> List[str]:
        return list(TRACE_TASKS.keys())
    
    def get_task_info(self, task_id: str) -> TRACETaskInfo:
        if task_id not in TRACE_TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        return TRACE_TASKS[task_id]
    
    def load_task(self, task_id: str):
        if task_id in self._datasets:
            logger.info(f"Task {task_id} already loaded, skipping")
            return
        
        logger.info(f"Loading task: {task_id}")
        self._datasets[task_id] = {}
    
        self._datasets[task_id]['train'] = TRACEDataset(
            task_id=task_id,
            split='train',
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            data_path=self.data_path,
            max_samples=self.train_samples,
            for_training=True,
            min_answer_tokens=self.min_answer_tokens,
        )

        for split, max_samples in [('eval', self.eval_samples), ('test', self.test_samples)]:
            self._datasets[task_id][split] = TRACEDataset(
                task_id=task_id,
                split=split,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                data_path=self.data_path,
                max_samples=max_samples,
                for_training=False,
                min_answer_tokens=self.min_answer_tokens,
            )
        
        logger.info(f"Task {task_id} loaded successfully")
    
    def get_train_dataloader(self, task_id: str) -> DataLoader:
        return DataLoader(
            self._datasets[task_id]['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # decide how to combine the samples in each batch.
            collate_fn=train_collate_fn,
            drop_last=True, 
        )
    
    def get_eval_dataloader(self, task_id: str, split: str = 'eval') -> DataLoader:
        tokenizer = self.tokenizer
        collate = lambda batch: eval_collate_fn(batch, tokenizer)
        
        return DataLoader(
            self._datasets[task_id][split],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    def get_all_dataloaders(self, task_id: str) -> Dict[str, DataLoader]:
        return {
            'train': self.get_train_dataloader(task_id),
            'eval': self.get_eval_dataloader(task_id, 'eval'),
            'test': self.get_eval_dataloader(task_id, 'test'),
        }
