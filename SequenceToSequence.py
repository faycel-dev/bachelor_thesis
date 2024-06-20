import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        speaker_ids = [feature["speaker_id"] for feature in features]
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch_input = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == torch.tensor([50258])).all().cpu().item():
            labels = labels[:, 1:]
        
        # Construct the batch dictionary
        batch = {
            "input_features": batch_input["input_features"],  # Use batch_input here
            "labels": labels,
            "speaker_ids": torch.tensor(speaker_ids)}

        return batch
