import subprocess
import sys

def run_command(command):
    """
    Thực thi một lệnh và in ra output. 
    Thoát khỏi chương trình nếu lệnh thất bại.
    """
    print("-" * 60)
    print(f"▶️  Đang thực thi: {' '.join(command)}")
    print("-" * 60)
    try:
        subprocess.run(command, check=True, text=True, encoding='utf-8')
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy lệnh. '{command[0]}' đã được cài đặt và nằm trong PATH chưa?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI: Lệnh {' '.join(command)} thất bại với mã lỗi {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi không mong muốn: {e}")
        sys.exit(1)

def main():
    """Hàm chính điều phối toàn bộ quy trình."""

    # --- BƯỚC 1: FINE-TUNE CÁC MÔ HÌNH HỖ TRỢ ---
    print("\n🚀 BƯỚC 1: Bắt đầu Fine-tune các mô hình...")

    # 1.1 Fine-tune CLIP
    print("\n--- 1.1. Fine-tuning CLIP ---")
    # run_command([
    #     "python", "fine_tune_vision_language_model.py",
    #     "--model_type", "clip",
    #     "--output_dir", "./clip_finetuned_coco",
    #     "--epochs", "2",
    #     "--learning_rate", "1e-6",
    #     # THAY ĐỔI: Tăng batch size ở đây
    #     "--batch_size", "2" 
    # ])
    print("✅ Fine-tune CLIP hoàn tất!")

    # 1.2 Fine-tune SigLIP
    print("\n--- 1.2. Fine-tuning SigLIP ---")
    run_command([
        "python", "fine_tune_vision_language_model.py",
        "--model_type", "siglip",
        "--output_dir", "./siglip_finetuned_coco",
        "--epochs", "4",
        "--learning_rate", "8e-7",
        # THAY ĐỔI: Tăng batch size ở đây
        "--batch_size", "32",
        "--max_text_length", "64"  # Add this argument
    ])
    print("✅ Fine-tune SigLIP hoàn tất!")

    # --- BƯỚC 2: SINH CAPTION CHO BỘ DỮ LIỆU TEST ---
    # print("\n📝 BƯỚC 2: Bắt đầu sinh caption cho bộ dữ liệu Test...")
    # run_command(["python", "generate_captions.py"])
    # print("✅ Sinh caption hoàn tất!")

    # --- BƯỚC 3: LẬP CHỈ MỤC (INDEXING) ---
    # BLIP-2 sẽ sử dụng model gốc vì không được fine-tune ở bước 1.
    # CLIP và SigLIP sẽ sử dụng model đã fine-tune (nhờ cập nhật ở feature_extractor.py).
    # models_to_process = ["clip", "siglip", "blip"]
    
    # print("\n🖼️  BƯỚC 3: Bắt đầu lập chỉ mục (indexing)...")
    # for model_name in models_to_process:
    #     print(f"\n--- Đang lập chỉ mục với {model_name.upper()} ---")
    #     run_command(["python", "index.py", "--model", model_name])
    #     print(f"✅ Lập chỉ mục {model_name.upper()} hoàn tất!")

    # # --- BƯỚC 4: ĐÁNH GIÁ (EVALUATION) HIỆU NĂNG ---
    # print("\n📊 BƯỚC 4: Bắt đầu đánh giá hiệu năng...")
    # for model_name in models_to_process:
    #     print(f"\n--- Đang đánh giá {model_name.upper()} ---")
    #     run_command(["python", "evaluate.py", "--extractor", model_name])
    #     print(f"✅ Đánh giá {model_name.upper()} hoàn tất!")

    # --- KẾT THÚC ---
    print("\n==========================================================")
    print("🎉 TẤT CẢ CÁC QUY TRÌNH ĐÃ HOÀN TẤT! 🎉")
    print("Kiểm tra các file kết quả: result_clip.json, result_siglip.json, và result_blip.json")
    print("==========================================================")

if __name__ == "__main__":
    main()