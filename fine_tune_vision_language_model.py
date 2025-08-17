# fine_tune_vision_language_model.py (Đã tối ưu cho RTX 4090)
import json
import os
from PIL import Image
from datasets import Dataset
from transformers import (
    Trainer, TrainingArguments,
    default_data_collator,
    CLIPModel, CLIPProcessor,
    SiglipModel, SiglipProcessor
)
import torch
from tqdm import tqdm
import argparse
import transformers

MAX_TEXT_LENGTH = 77

class SigLipTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # Chỉ yêu cầu các trường cần thiết dựa trên mô hình
            required_keys = ["input_ids", "pixel_values"]
            if not all(key in inputs for key in required_keys):
                raise ValueError(f"Inputs thiếu các trường cần thiết: {list(inputs.keys())}")
            
            outputs = model(**inputs, return_loss=True)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                raise ValueError(f"Mô hình không trả về loss. Outputs: {list(outputs.keys())}")
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            print(f"LỖI trong compute_loss: {e}")
            raise

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs, return_loss=True)
        loss = outputs.loss.detach() if hasattr(outputs, "loss") else None
        logits = outputs.logits_per_image
        labels = None
        return (loss, logits, labels)

# --- Cấu hình các Model được hỗ trợ ---
SUPPORTED_MODELS = {
    "clip": {
        "model_class": CLIPModel,
        "processor_class": CLIPProcessor,
        "default_name": "openai/clip-vit-large-patch14",
        "requires_attention_mask": True
    },
    "siglip": {
        "model_class": SiglipModel,
        "processor_class": SiglipProcessor,
        "default_name": "google/siglip-large-patch16-384",
        "requires_attention_mask": False
    }
}

def create_coco_dataset(annotations_file, images_dir):
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Không tìm thấy file annotations: {annotations_file}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    with open(annotations_file, "r") as f:
        data = json.load(f)
    image_info = {img["id"]: img["file_name"] for img in data["images"]}
    ann_info = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_info:
            ann_info[img_id] = []
        ann_info[img_id].append(ann["caption"])
    dataset_list = []
    for img_id, captions in tqdm(ann_info.items(), desc="Đang xử lý dataset"):
        file_path = os.path.join(images_dir, image_info[img_id])
        if os.path.exists(file_path):
            dataset_list.append({"file_path": file_path, "caption": captions[0]})
    if not dataset_list:
        raise ValueError("Không tìm thấy ảnh hợp lệ nào trong dataset.")
    dataset = Dataset.from_list(dataset_list)
    print(f"Dataset có {len(dataset)} mẫu.")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune một mô hình Vision-Language trên COCO.")
    parser.add_argument("--model_type", type=str, required=True, choices=["clip", "siglip"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch. Tăng lên 32 hoặc 64 nếu VRAM cho phép.")
    parser.add_argument("--max_text_length", type=int, default=77)
    args = parser.parse_args()

    model_config = SUPPORTED_MODELS[args.model_type]
    ModelClass = model_config["model_class"]
    ProcessorClass = model_config["processor_class"]
    model_name = args.model_name if args.model_name else model_config["default_name"]
    requires_attention_mask = model_config["requires_attention_mask"]
    
    ANNOTATIONS_FILE = "annotations/captions_val2017.json"
    IMAGES_DIR = "dataset/"

    processor = ProcessorClass.from_pretrained(model_name)

    def preprocess_function(examples):
        try:
            images = [Image.open(file_path).convert("RGB") for file_path in examples["file_path"]]
            texts = examples["caption"] if isinstance(examples["caption"], list) else [examples["caption"]]
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_text_length
            )
            # Đảm bảo tất cả trường là tensor
            for key in inputs:
                if isinstance(inputs[key], list):
                    inputs[key] = torch.tensor(inputs[key], dtype=torch.long if key in ["input_ids", "attention_mask"] else torch.float)
            return inputs
        except Exception as e:
            print(f"LỖI: Không thể preprocess batch: {e}")
            return None

    dataset = create_coco_dataset(ANNOTATIONS_FILE, IMAGES_DIR)
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    
    # Preprocess dataset
    print("Bắt đầu preprocess dataset...")
    try:
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=args.batch_size,
            num_proc=1,  # Tắt multiprocessing để tránh lỗi trên Windows
            remove_columns=["file_path", "caption"]
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=args.batch_size,
            num_proc=1,
            remove_columns=["file_path", "caption"]
        )
    except Exception as e:
        print(f"LỖI: Không thể preprocess dataset: {e}")
        exit(1)

    # Lọc các mẫu None
    print("Lọc các mẫu không hợp lệ...")
    train_dataset = train_dataset.filter(
        lambda x: x is not None and all(k in x for k in ["input_ids", "pixel_values"]),
        num_proc=1
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x is not None and all(k in x for k in ["input_ids", "pixel_values"]),
        num_proc=1
    )

    # Định dạng dataset
    print("Định dạng dataset thành PyTorch Tensors...")
    columns = ["input_ids", "pixel_values"]
    if requires_attention_mask:
        columns.append("attention_mask")
    train_dataset.set_format(type="torch", columns=columns)
    eval_dataset.set_format(type="torch", columns=columns)

    # Kiểm tra dataset
    print(f"Dataset sau preprocess và lọc: train={len(train_dataset)} mẫu, eval={len(eval_dataset)} mẫu.")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if all(k in sample for k in columns):
            print(f"Mẫu train {i}: input_ids shape={sample['input_ids'].shape}, pixel_values shape={sample['pixel_values'].shape}")
            if requires_attention_mask:
                print(f"  attention_mask shape={sample['attention_mask'].shape}")
        else:
            print(f"Mẫu train {i}: Dữ liệu không hợp lệ.")

    print(f"Đang tải model {model_name}...")
    model = ModelClass.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        bf16=True,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        overwrite_output_dir=True,
        report_to="none"
    )

    trainer = SigLipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Bắt đầu quá trình fine-tune đã tối ưu...")
    trainer.train()
    print("🎉 Fine-tune hoàn tất!")

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("✅ Đã lưu thành công model tốt nhất!")

if __name__ == "__main__":
    main()