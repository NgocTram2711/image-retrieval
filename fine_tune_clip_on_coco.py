# fine_tune_clip_on_coco.py
import json
import os
from PIL import Image
from datasets import Dataset
from transformers import CLIPProcessor, CLIPModel, Trainer, TrainingArguments
from transformers import default_data_collator
import torch
from tqdm import tqdm
import transformers

# Kiểm tra phiên bản transformers
print(f"Phiên bản transformers: {transformers.__version__}")
if transformers.__version__ < "4.30.0":
    print("CẢNH BÁO: Phiên bản transformers quá cũ. Vui lòng nâng cấp lên >=4.30.0 bằng lệnh: pip install transformers --upgrade")
    exit(1)

# --- Cấu hình ---
MODEL_NAME = "openai/clip-vit-large-patch14"
ANNOTATIONS_FILE = "annotations/captions_val2017.json"
IMAGES_DIR = "dataset/"
OUTPUT_DIR = "./clip_finetuned_coco"
NUM_EPOCHS = 2
LEARNING_RATE = 1e-6
BATCH_SIZE = 2
MAX_TEXT_LENGTH = 77

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # Đảm bảo inputs chứa các trường cần thiết
            required_keys = ["input_ids", "attention_mask", "pixel_values"]
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
        else:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {file_path}")
    if not dataset_list:
        raise ValueError("Không tìm thấy ảnh hợp lệ nào trong dataset.")
    dataset = Dataset.from_list(dataset_list)
    print(f"Dataset có {len(dataset)} mẫu.")
    return dataset

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

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
            max_length=MAX_TEXT_LENGTH
        )
        # Đảm bảo tất cả trường là tensor
        for key in inputs:
            if isinstance(inputs[key], list):
                inputs[key] = torch.tensor(inputs[key], dtype=torch.long if key in ["input_ids", "attention_mask"] else torch.float)
        return inputs
    except Exception as e:
        print(f"LỖI: Không thể preprocess batch: {e}")
        return None

if __name__ == "__main__":
    # Tạo dataset
    try:
        dataset = create_coco_dataset(ANNOTATIONS_FILE, IMAGES_DIR)
    except Exception as e:
        print(f"LỖI: Không thể tạo dataset: {e}")
        exit(1)

    # Tách dataset
    print("Tách dataset thành tập train và validation...")
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    print(f"Tập train: {len(train_dataset)} mẫu, tập validation: {len(eval_dataset)} mẫu.")

    # Preprocess dataset
    print("Bắt đầu preprocess dataset...")
    try:
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=1,  # Tắt multiprocessing để tránh lỗi trên Windows
            remove_columns=["file_path", "caption"]
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=1,
            remove_columns=["file_path", "caption"]
        )
    except Exception as e:
        print(f"LỖI: Không thể preprocess dataset: {e}")
        exit(1)

    # Lọc các mẫu None
    print("Lọc các mẫu không hợp lệ...")
    train_dataset = train_dataset.filter(
        lambda x: x is not None and all(k in x for k in ["input_ids", "pixel_values", "attention_mask"]),
        num_proc=1
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x is not None and all(k in x for k in ["input_ids", "pixel_values", "attention_mask"]),
        num_proc=1
    )

    # Định dạng dataset
    print("Định dạng dataset thành PyTorch Tensors...")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values"])

    # Kiểm tra dataset
    print(f"Dataset sau preprocess và lọc: train={len(train_dataset)} mẫu, eval={len(eval_dataset)} mẫu.")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if all(k in sample for k in ["input_ids", "pixel_values", "attention_mask"]):
            print(f"Mẫu train {i}: input_ids shape={sample['input_ids'].shape}, pixel_values shape={sample['pixel_values'].shape}")
        else:
            print(f"Mẫu train {i}: Dữ liệu không hợp lệ.")

    # Load model
    try:
        print(f"Đang tải model {MODEL_NAME}...")
        model = CLIPModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"LỖI: Không thể tải model {MODEL_NAME}: {e}")
        exit(1)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Sửa thành eval_loss
        save_total_limit=3,
        overwrite_output_dir=True,
        report_to="none"
    )

    # Trainer
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # Fine-tune
    try:
        print("Bắt đầu fine-tune model...")
        trainer.train()
        print("🎉 Fine-tune hoàn tất!")
    except Exception as e:
        print(f"LỖI trong quá trình fine-tune: {e}")
        exit(1)

    # Lưu model
    try:
        print(f"Đang lưu model và processor vào {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print("✅ Đã lưu thành công model tốt nhất!")
    except Exception as e:
        print(f"LỖI: Không thể lưu model: {e}")
        exit(1)