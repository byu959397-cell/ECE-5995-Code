import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    task_id: str
    accuracy: float = 0.0
    rougeL: float = 0.0
    loss: float = 0.0
    num_samples: int = 0
    num_correct: int = 0
    extraction_success: int = 0  # 成功提取答案的数量
    extraction_rate: float = 1.0  # 答案提取成功率
    sample_preds: List[str] = field(default_factory=list)
    sample_labels: List[str] = field(default_factory=list)


@dataclass
class ContinualMetrics:
    average_accuracy: float = 0.0
    backward_transfer: float = 0.0
    forgetting: float = 0.0
    forward_transfer: float = 0.0
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)
    accuracy_matrix: Optional[np.ndarray] = None

class AnswerExtractor:
    # text answer appeared in NumGLUE-cm
    MONTHS = frozenset([
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ])
    
    WEEKDAYS = frozenset([
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    ])
    
    KNOWN_NAMES = frozenset(['mia', 'cameron', 'julian'])
    
    ALL_TEXT_ANSWERS = MONTHS | WEEKDAYS | KNOWN_NAMES
    
    @staticmethod
    def extract_choice_letter(
        text: str, 
        valid_options: str = "ABC",
        strict_first: bool = True
    ) -> str:
        if not text:
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        valid_set = set(valid_options.upper())
        valid_pattern = ''.join(sorted(valid_set))
        
        structured_patterns = [
            # Chinese text
            rf'(?:答案|选择|选项|结果)\s*(?:是|为|:|：)\s*([{valid_pattern}])\b',
            rf'(?:应该选|我选|选)\s*([{valid_pattern}])\b',
            rf'正确(?:答案|选项)(?:是|为|:|：)?\s*([{valid_pattern}])\b',
            
            rf'(?:answer|choice|option)\s*(?:is|:)\s*([{valid_pattern}])\b',
            rf'(?:the answer is|i choose|i pick|i select|i would choose|i\'d choose)\s*([{valid_pattern}])\b',
            rf'(?:correct answer|best answer|right answer)\s*(?:is|:)?\s*([{valid_pattern}])\b',
            
            rf'\b([{valid_pattern}])\s+(?:is correct|is the answer|is right|是正确的|是答案)',
            
            rf'(?:choose|select|pick)\s+([{valid_pattern}])\b',
        ]
        
        for pattern in structured_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        if len(text) == 1:
            c = text.upper()
            return c if c in valid_set else ""
        
        first_char = text[0].upper()
        if first_char in valid_set:
            if len(text) == 1:
                return first_char
            second_char = text[1]
            if second_char in '.,:;)]\t\n 、。．：':
                return first_char
            if not strict_first:
                return first_char
        
        first_line = text.split('\n')[0].strip()
        if first_line:
            first_line_clean = re.sub(r'^[\s\-\*\#\>\•\[\]]+', '', first_line).strip()
            if first_line_clean:
                c = first_line_clean[0].upper()
                if c in valid_set:
                    if len(first_line_clean) == 1:
                        return c
                    if first_line_clean[1] in '.,:;)]\t\n 、。．：':
                        return c
        
        choice_ref = rf'(?:选\s*([{valid_pattern}])|([{valid_pattern}])\s*选项)'
        match = re.search(choice_ref, text, re.IGNORECASE)
        if match:
            return (match.group(1) or match.group(2)).upper()
        
        bracket_pattern = r'[\(\[]([' + valid_pattern + r'])[\)\]]'
        match = re.search(bracket_pattern, text)
        if match:
            return match.group(1).upper()
        
        independent_pattern = rf'(?<![a-zA-Z])([{valid_pattern}])(?![a-zA-Z])'
        match = re.search(independent_pattern, text)
        if match:
            return match.group(1).upper()
        
        return ""
    
    @staticmethod
    def detect_options_from_prompt(prompt: str) -> str:
        found_options = set()
        matches = re.findall(r'\n\s*([A-D])\s*[\.\:：]', prompt)
        found_options.update(matches)
        matches = re.findall(r'[\(\[]([A-D])[\)\]]', prompt)
        found_options.update(matches)
        matches = re.findall(r'([A-D])[、．]', prompt)
        found_options.update(matches)
        
        if found_options:
            return ''.join(sorted(found_options))
        
        return "ABC"
    
    @staticmethod
    def extract_cstance_answer(model_output: str) -> str:
        return AnswerExtractor.extract_choice_letter(
            model_output, 
            valid_options="ABC",
            strict_first=True
        )
    
    @staticmethod
    def extract_fomc_answer(model_output: str) -> str:
        return AnswerExtractor.extract_choice_letter(
            model_output,
            valid_options="ABC", 
            strict_first=True
        )
    
    @staticmethod
    def extract_scienceqa_answer(model_output: str, prompt: Optional[str] = None) -> str:
        if prompt:
            valid_options = AnswerExtractor.detect_options_from_prompt(prompt)
        else:
            valid_options = "ABCD"
        
        return AnswerExtractor.extract_choice_letter(
            model_output,
            valid_options=valid_options,
            strict_first=True
        )
    
    @staticmethod
    def extract_ground_truth_choice(answer_field: str) -> str:
        if not answer_field:
            return ""
        
        answer_field = answer_field.strip()
        if not answer_field:
            return ""
        
        first_char = answer_field[0].upper()
        return first_char if first_char in 'ABCD' else ""
    
    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        if not text:
            return None
        
        text = text.strip()
        if not text:
            return None
        
        cleaned = text.replace(',', '').replace('，', '')
        
        try:
            return float(cleaned)
        except ValueError:
            pass
        
        number_pattern = r'[-+]?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def extract_text_answer(text: str) -> Optional[str]:
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        # 在文本中查找已知的文本答案
        for keyword in AnswerExtractor.ALL_TEXT_ANSWERS:
            # 使用词边界匹配，避免部分匹配（如 "may" 在 "maybe" 中）
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, text_lower):
                return keyword.capitalize()
        
        return None
    
    @staticmethod
    def extract_numglue_cm_answer(text: str) -> Tuple[Optional[float], Optional[str]]:
        if not text:
            return None, None
        
        num = AnswerExtractor.extract_number(text)
        if num is not None:
            return num, None
        
        text_ans = AnswerExtractor.extract_text_answer(text)
        if text_ans is not None:
            return None, text_ans
        
        return None, None
    
    @staticmethod
    def extract_numglue_ds_answer(text: str) -> Optional[float]:
        return AnswerExtractor.extract_number(text)


class AnswerMatcher:
    @staticmethod
    def match_choice(pred: str, ref: str) -> bool:
        return pred.upper() == ref.upper() if pred and ref else False
    
    @staticmethod
    def match_number_exact(pred: Optional[float], ref: Optional[float]) -> bool:
        if pred is None or ref is None:
            return False
        return abs(pred - ref) < 0.5
    
    @staticmethod
    def match_number_tolerance(
        pred: Optional[float], 
        ref: Optional[float],
        tolerance: float = 0.02
    ) -> bool:
        if pred is None or ref is None:
            return False
        
        if pred == ref:
            return True
        
        if ref != 0:
            rel_error = abs(pred - ref) / abs(ref)
            return rel_error <= tolerance
        
        return abs(pred) < 1e-6
    
    @staticmethod
    def match_text_ignore_case(pred: Optional[str], ref: Optional[str]) -> bool:
        if pred is None or ref is None:
            return False
        return pred.lower().strip() == ref.lower().strip()
    
    @staticmethod
    def match_numglue_cm(pred_text: str, ref_text: str) -> bool:
        ref_text = ref_text.strip()
        
        try:
            ref_num = float(ref_text.replace(',', ''))
            is_ref_numeric = True
        except ValueError:
            is_ref_numeric = False
        
        if is_ref_numeric:
            pred_num = AnswerExtractor.extract_number(pred_text)
            if pred_num is None:
                return False
            
            if '.' not in ref_text:
                return AnswerMatcher.match_number_exact(pred_num, ref_num)
            else:
                return AnswerMatcher.match_number_tolerance(pred_num, ref_num, tolerance=0.01)
        else:
            pred_text_ans = AnswerExtractor.extract_text_answer(pred_text)
            if pred_text_ans is None:
                return AnswerMatcher.match_text_ignore_case(pred_text.strip(), ref_text)
            return AnswerMatcher.match_text_ignore_case(pred_text_ans, ref_text)
    
    @staticmethod
    def match_numglue_ds(pred_text: str, ref_text: str) -> bool:
        ref_text = ref_text.strip()
        
        pred_num = AnswerExtractor.extract_number(pred_text)
        ref_num = AnswerExtractor.extract_number(ref_text)
        
        if pred_num is None or ref_num is None:
            return False
        
        if '.' not in ref_text:
            return AnswerMatcher.match_number_exact(pred_num, ref_num)
        else:
            return AnswerMatcher.match_number_tolerance(pred_num, ref_num, tolerance=0.02)


class MetricsCalculator:
    def __init__(self):
        self._rouge_scorer = None
    
    @property
    def rouge_scorer(self):
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rougeL'], 
                    use_stemmer=True
                )
            except ImportError:
                logger.warning(
                    "rouge_score not installed. "
                    "Install with: pip install rouge-score"
                )
        return self._rouge_scorer

    def compute_rouge_l(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> float:
        if self.rouge_scorer is None:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            if pred and ref:
                try:
                    score = self.rouge_scorer.score(ref, pred)
                    scores.append(score['rougeL'].fmeasure)
                except Exception as e:
                    logger.debug(f"ROUGE error: {e}")
                    scores.append(0.0)
            else:
                scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0

    def compute_trace_metrics(
        self, 
        predictions: List[str], 
        references: List[str], 
        task_id: str,
        prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        task_lower = task_id.lower().replace('-', '_')
        n = len(references)
        
        if n == 0:
            return {
                "accuracy": 0.0, "rougeL": 0.0,
                "num_correct": 0, "extraction_success": 0
            }
        if 'c_stance' in task_lower or 'cstance' in task_lower:
            correct, extracted = 0, 0
            for pred, ref in zip(predictions, references):
                pred_opt = AnswerExtractor.extract_cstance_answer(pred)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                if pred_opt:
                    extracted += 1
                if AnswerMatcher.match_choice(pred_opt, ref_opt):
                    correct += 1
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }

        elif 'fomc' in task_lower:
            correct, extracted = 0, 0
            for pred, ref in zip(predictions, references):
                pred_opt = AnswerExtractor.extract_fomc_answer(pred)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                if pred_opt:
                    extracted += 1
                if AnswerMatcher.match_choice(pred_opt, ref_opt):
                    correct += 1
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }

        elif 'scienceqa' in task_lower or 'science_qa' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                prompt = prompts[i] if prompts and i < len(prompts) else None
                pred_opt = AnswerExtractor.extract_scienceqa_answer(pred, prompt)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                if pred_opt:
                    extracted += 1
                if AnswerMatcher.match_choice(pred_opt, ref_opt):
                    correct += 1
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }

        elif 'numglue_cm' in task_lower or 'numglue-cm' in task_lower:
            correct, extracted = 0, 0
            for pred, ref in zip(predictions, references):
                pred_num, pred_txt = AnswerExtractor.extract_numglue_cm_answer(pred)
                if pred_num is not None or pred_txt is not None:
                    extracted += 1
                if AnswerMatcher.match_numglue_cm(pred, ref):
                    correct += 1
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }

        elif 'numglue_ds' in task_lower or 'numglue-ds' in task_lower:
            correct, extracted = 0, 0
            for pred, ref in zip(predictions, references):
                pred_num = AnswerExtractor.extract_numglue_ds_answer(pred)
                if pred_num is not None:
                    extracted += 1
                if AnswerMatcher.match_numglue_ds(pred, ref):
                    correct += 1
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }

        elif 'meeting_bank' in task_lower or 'meetingbank' in task_lower:
            rouge = self.compute_rouge_l(predictions, references)
            return {
                "accuracy": 0.0, "rougeL": rouge,
                "num_correct": 0, "extraction_success": n
            }

        elif 'py150' in task_lower:
            rouge = self.compute_rouge_l(predictions, references)
            return {
                "accuracy": 0.0, "rougeL": rouge,
                "num_correct": 0, "extraction_success": n
            }

        elif '20minuten' in task_lower:
            rouge = self.compute_rouge_l(predictions, references)
            return {
                "accuracy": 0.0, "rougeL": rouge,
                "num_correct": 0, "extraction_success": n
            }
        
        else:
            logger.warning(f"Unknown task: {task_id}, using default evaluation")
            correct = 0
            for pred, ref in zip(predictions, references):
                pred_opt = AnswerExtractor.extract_choice_letter(pred)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                if AnswerMatcher.match_choice(pred_opt, ref_opt):
                    correct += 1
            
            if correct > 0:
                return {
                    "accuracy": correct / n, "rougeL": 0.0,
                    "num_correct": correct, "extraction_success": n
                }
            
            rouge = self.compute_rouge_l(predictions, references)
            return {
                "accuracy": 0.0, "rougeL": rouge,
                "num_correct": 0, "extraction_success": n
            }


class TRACEEvaluator:
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: torch.device,
        max_new_tokens: int = 256,
        log_dir: Optional[str] = None  
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.calc = MetricsCalculator()
        self.log_dir = log_dir  
        
        self._metric_matrix: Optional[np.ndarray] = None
        self._task_order: List[str] = []
        self._n_tasks: int = 0
    
    def set_log_dir(self, log_dir: str):
        """设置日志目录"""
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)

    def evaluate_task(
        self,
        task_id: str,
        dataloader,
        max_new_tokens: Optional[int] = None,
        current_task_idx: Optional[int] = None,
        save_log: bool = True
    ) -> TaskMetrics:
        self.model.eval()
        
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        all_predictions = []
        all_references = []
        all_prompts = []
        total_loss = 0.0
        num_batches = 0
        detailed_logs = []
        
        logger.info(f"Evaluating task: {task_id}")
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                raw_prompts = batch.get('raw_prompt', None)
                raw_answers = batch.get('raw_answer', None)
                
                batch = {
                    k: v.to(self.device) 
                    for k, v in batch.items() 
                    if isinstance(v, torch.Tensor)
                }
                
                try:
                    outputs = self.model(**batch)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss_val = outputs.loss.item()
                        if not np.isnan(loss_val):
                            total_loss += loss_val
                            num_batches += 1
                except Exception as e:
                    logger.debug(f"Loss computation failed: {e}")
                
                try:
                    gen_outputs = self.model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    continue
                
                input_len = batch['input_ids'].shape[1]
                for i in range(len(batch['input_ids'])):
                    if gen_outputs.shape[1] > input_len:
                        pred_ids = gen_outputs[i, input_len:]
                        pred_text = self.tokenizer.decode(
                            pred_ids, skip_special_tokens=True
                        ).strip()
                    else:
                        pred_text = ""
                    
                    ref_text = raw_answers[i] if raw_answers else ""
                    prompt_text = raw_prompts[i] if raw_prompts else ""
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)
                    all_prompts.append(prompt_text)
        
        print(f"\n[Evaluation Samples - {task_id}]")
        for i in range(min(3, len(all_predictions))):
            print(f"  Pred : {repr(all_predictions[i][:80])}")
            print(f"  Label: {repr(all_references[i][:80])}")
            print("-" * 50)
        
        scores, detailed_logs = self._compute_metrics_with_details(
            all_predictions, all_references, task_id, all_prompts
        )
        
        metrics = TaskMetrics(task_id=task_id)
        metrics.num_samples = len(all_predictions)
        metrics.accuracy = scores.get("accuracy", 0.0)
        metrics.rougeL = scores.get("rougeL", 0.0)
        metrics.num_correct = scores.get("num_correct", 0)
        metrics.extraction_success = scores.get("extraction_success", 0)
        metrics.extraction_rate = metrics.extraction_success / max(metrics.num_samples, 1)
        metrics.loss = total_loss / max(num_batches, 1)
        metrics.sample_preds = all_predictions[:5]
        metrics.sample_labels = all_references[:5]
        
        if save_log and self.log_dir:
            self._save_evaluation_log(
                task_id, current_task_idx, metrics, detailed_logs
            )
        if metrics.accuracy > 0:
            print(f"  Accuracy: {metrics.accuracy:.4f} ({metrics.num_correct}/{metrics.num_samples})")
            print(f"  Extraction Rate: {metrics.extraction_rate:.2%}")
        else:
            print(f"  ROUGE-L: {metrics.rougeL:.4f}")
        
        return metrics
    
    def _compute_metrics_with_details(
        self,
        predictions: List[str],
        references: List[str],
        task_id: str,
        prompts: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        task_lower = task_id.lower().replace('-', '_')
        n = len(references)
        detailed_logs = []
        
        if n == 0:
            return {"accuracy": 0.0, "rougeL": 0.0, "num_correct": 0, "extraction_success": 0}, []

        # =====================================================================
        # C-STANCE
        # =====================================================================
        if 'c_stance' in task_lower or 'cstance' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_opt = AnswerExtractor.extract_cstance_answer(pred)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                is_correct = AnswerMatcher.match_choice(pred_opt, ref_opt)
                if pred_opt:
                    extracted += 1
                if is_correct:
                    correct += 1
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompts[i] if prompts else "",
                    "reference": ref,
                    "reference_extracted": ref_opt,
                    "prediction": pred,
                    "prediction_extracted": pred_opt,
                    "is_correct": is_correct,
                    "extraction_success": bool(pred_opt)
                })
            
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }, detailed_logs

        # =====================================================================
        # FOMC
        # =====================================================================
        elif 'fomc' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_opt = AnswerExtractor.extract_fomc_answer(pred)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                is_correct = AnswerMatcher.match_choice(pred_opt, ref_opt)
                if pred_opt:
                    extracted += 1
                if is_correct:
                    correct += 1
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompts[i] if prompts else "",
                    "reference": ref,
                    "reference_extracted": ref_opt,
                    "prediction": pred,
                    "prediction_extracted": pred_opt,
                    "is_correct": is_correct,
                    "extraction_success": bool(pred_opt)
                })
            
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }, detailed_logs

        # =====================================================================
        # ScienceQA
        # =====================================================================
        elif 'scienceqa' in task_lower or 'science_qa' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                prompt = prompts[i] if prompts and i < len(prompts) else None
                valid_options = AnswerExtractor.detect_options_from_prompt(prompt) if prompt else "ABCD"
                pred_opt = AnswerExtractor.extract_scienceqa_answer(pred, prompt)
                ref_opt = AnswerExtractor.extract_ground_truth_choice(ref)
                is_correct = AnswerMatcher.match_choice(pred_opt, ref_opt)
                if pred_opt:
                    extracted += 1
                if is_correct:
                    correct += 1
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompt[:200] + "..." if prompt and len(prompt) > 200 else prompt,
                    "valid_options": valid_options,
                    "reference": ref[:100] + "..." if len(ref) > 100 else ref,
                    "reference_extracted": ref_opt,
                    "prediction": pred,
                    "prediction_extracted": pred_opt,
                    "is_correct": is_correct,
                    "extraction_success": bool(pred_opt)
                })
            
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }, detailed_logs

        # =====================================================================
        # NumGLUE-cm
        # =====================================================================
        elif 'numglue_cm' in task_lower or 'numglue-cm' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_num, pred_txt = AnswerExtractor.extract_numglue_cm_answer(pred)
                ref_num, ref_txt = AnswerExtractor.extract_numglue_cm_answer(ref)
                is_correct = AnswerMatcher.match_numglue_cm(pred, ref)
                has_extraction = pred_num is not None or pred_txt is not None
                if has_extraction:
                    extracted += 1
                if is_correct:
                    correct += 1
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompts[i] if prompts else "",
                    "reference": ref,
                    "reference_type": "number" if ref_num is not None else "text",
                    "reference_extracted": ref_num if ref_num is not None else ref_txt,
                    "prediction": pred,
                    "prediction_type": "number" if pred_num is not None else ("text" if pred_txt else "none"),
                    "prediction_extracted": pred_num if pred_num is not None else pred_txt,
                    "is_correct": is_correct,
                    "extraction_success": has_extraction
                })
            
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }, detailed_logs

        # =====================================================================
        # NumGLUE-ds
        # =====================================================================
        elif 'numglue_ds' in task_lower or 'numglue-ds' in task_lower:
            correct, extracted = 0, 0
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                pred_num = AnswerExtractor.extract_numglue_ds_answer(pred)
                ref_num = AnswerExtractor.extract_numglue_ds_answer(ref)
                is_correct = AnswerMatcher.match_numglue_ds(pred, ref)
                if pred_num is not None:
                    extracted += 1
                if is_correct:
                    correct += 1
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompts[i] if prompts else "",
                    "reference": ref,
                    "reference_extracted": ref_num,
                    "prediction": pred,
                    "prediction_extracted": pred_num,
                    "is_correct": is_correct,
                    "extraction_success": pred_num is not None,
                    "error_rate": abs(pred_num - ref_num) / abs(ref_num) if (pred_num and ref_num and ref_num != 0) else None
                })
            
            return {
                "accuracy": correct / n, "rougeL": 0.0,
                "num_correct": correct, "extraction_success": extracted
            }, detailed_logs

        # =====================================================================
        # MeetingBank / Py150 / 20Minuten (ROUGE-L)
        # =====================================================================
        elif any(x in task_lower for x in ['meeting_bank', 'meetingbank', 'py150', '20minuten']):
            rouge = self.calc.compute_rouge_l(predictions, references)
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                # 计算单个样本的ROUGE-L
                try:
                    if self.calc.rouge_scorer and pred and ref:
                        score = self.calc.rouge_scorer.score(ref, pred)['rougeL'].fmeasure
                    else:
                        score = 0.0
                except:
                    score = 0.0
                
                detailed_logs.append({
                    "index": i,
                    "prompt": prompts[i][:200] + "..." if prompts and len(prompts[i]) > 200 else (prompts[i] if prompts else ""),
                    "reference": ref[:300] + "..." if len(ref) > 300 else ref,
                    "prediction": pred[:300] + "..." if len(pred) > 300 else pred,
                    "rouge_l": score
                })
            
            return {
                "accuracy": 0.0, "rougeL": rouge,
                "num_correct": 0, "extraction_success": n
            }, detailed_logs

        else:
            logger.warning(f"Unknown task: {task_id}")
            return {"accuracy": 0.0, "rougeL": 0.0, "num_correct": 0, "extraction_success": 0}, []
    
    def _save_evaluation_log(
        self,
        task_id: str,
        current_task_idx: Optional[int],
        metrics: TaskMetrics,
        detailed_logs: List[Dict]
    ):
        import json
        import os
        from datetime import datetime
        
        if current_task_idx is not None:
            filename = f"eval_log_after_task{current_task_idx+1}_on_{task_id}.json"
        else:
            filename = f"eval_log_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        log_content = {
            "metadata": {
                "task_id": task_id,
                "current_task_idx": current_task_idx,
                "timestamp": datetime.now().isoformat(),
                "num_samples": metrics.num_samples,
            },
            "summary": {
                "accuracy": metrics.accuracy,
                "rouge_l": metrics.rougeL,
                "num_correct": metrics.num_correct,
                "extraction_success": metrics.extraction_success,
                "extraction_rate": metrics.extraction_rate,
                "loss": metrics.loss
            },
            "detailed_results": detailed_logs
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_content, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation log saved to: {filepath}")

    def evaluate_all_tasks(
        self, 
        dataloaders: Dict[str, Any], 
        current_task_idx: int
    ) -> Dict[str, TaskMetrics]:
        results = {}
        task_list = list(dataloaders.keys())
        n_tasks = len(task_list)
        
        if self._metric_matrix is None or self._n_tasks != n_tasks:
            self._metric_matrix = np.zeros((n_tasks, n_tasks))
            self._task_order = task_list
            self._n_tasks = n_tasks
        
        for t_idx in range(current_task_idx + 1):
            task_id = task_list[t_idx]
            metrics = self.evaluate_task(task_id, dataloaders[task_id])
            results[task_id] = metrics
            
            main_score = metrics.accuracy if metrics.accuracy > 0 else metrics.rougeL
            self._metric_matrix[current_task_idx, t_idx] = main_score
        
        return results

    def compute_continual_metrics(self) -> ContinualMetrics:
        if self._metric_matrix is None:
            return ContinualMetrics()
        
        matrix = self._metric_matrix
        n_tasks = matrix.shape[0]
        
        current_k = 0
        for i in range(n_tasks):
            if np.sum(matrix[i]) > 0:
                current_k = i
        
        # Average Accuracy
        final_scores = matrix[current_k, :current_k+1]
        avg_acc = float(np.mean(final_scores)) if len(final_scores) > 0 else 0.0
        
        # Forgetting
        forgetting = 0.0
        if current_k > 0:
            for j in range(current_k):
                max_score = np.max(matrix[:current_k+1, j])
                current_score = matrix[current_k, j]
                forgetting += max(0, max_score - current_score)
            forgetting /= current_k
        
        # Backward Transfer
        bwt = 0.0
        if current_k > 0:
            for j in range(current_k):
                initial_score = matrix[j, j]
                current_score = matrix[current_k, j]
                bwt += (current_score - initial_score)
            bwt /= current_k

        return ContinualMetrics(
            average_accuracy=avg_acc,
            forgetting=float(forgetting),
            backward_transfer=float(bwt),
            accuracy_matrix=matrix.copy()
        )

    def print_results(
        self, 
        continual_metrics: ContinualMetrics, 
        current_results: Dict[str, TaskMetrics]
    ):
        print("\n" + "=" * 75)
        print("TRACE Benchmark Evaluation Report")
        print("=" * 75)
        
        print(f"\nContinual Learning Metrics:")
        print(f"  Average Score:       {continual_metrics.average_accuracy:.4f}")
        print(f"  Forgetting:          {continual_metrics.forgetting:.4f} (↓ lower is better)")
        print(f"  Backward Transfer:   {continual_metrics.backward_transfer:+.4f}")
        
        print(f"\nPer-Task Results:")
        header = f"{'Task':<18} | {'Metric':<10} | {'Score':<8} | {'Correct':<12} | {'Extract%':<10}"
        print(header)
        print("-" * 75)
        
        for task_id, m in current_results.items():
            if m.rougeL > 0 and m.accuracy == 0:
                metric_name, metric_val = "ROUGE-L", m.rougeL
                correct_str = "N/A"
            else:
                metric_name, metric_val = "Accuracy", m.accuracy
                correct_str = f"{m.num_correct}/{m.num_samples}"
            
            extract_str = f"{m.extraction_rate:.1%}"
            print(f"{task_id:<18} | {metric_name:<10} | {metric_val:.4f}   | {correct_str:<12} | {extract_str:<10}")
        

