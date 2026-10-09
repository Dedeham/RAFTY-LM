import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", str(Path(".triton_cache").resolve()))
os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
Path(".triton_cache").mkdir(exist_ok=True)
Path(".hf_cache").mkdir(exist_ok=True)

from src.config import load_config

TRAIN_DATA_PATH = Path("data/raft_train.json")
LORA_OUTPUT_DIR = Path("models/raft_lora")
GGUF_OUTPUT_DIR = Path("models/raft_gguf")


def build_system_prompt(domain_label):
    return f"""You are a certified expert in {domain_label}, trained in the RAFT (Retrieval-Augmented Fine-Tuning) methodology.
Given a user question and candidate context documents:
1. In your chain_of_thought, examine each document, explain why distractors are irrelevant, and identify golden evidence if present.
2. Set status to "answer_found" if the context contains the answer, or "all_distractors" if none of the documents contain the specification.
3. Provide the precise technical answer.
Respond strictly in JSON format matching the schema:
{{
  "chain_of_thought": "...",
  "status": "answer_found" | "all_distractors",
  "answer": "..."
}}"""


def format_prompt(sample, tokenizer, system_prompt):
    context_str = "\n\n".join(sample["context"])
    user_content = f"Question: {sample['question']}\n\nContext Documents:\n{context_str}"

    assistant_dict = {
        "chain_of_thought": sample["chain_of_thought"],
        "status": sample["status"],
        "answer": sample["answer"]
    }
    assistant_content = json.dumps(assistant_dict, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prepare_hf_dataset(data_path, tokenizer, system_prompt):
    from datasets import Dataset

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    formatted_texts = [format_prompt(s, tokenizer, system_prompt) for s in samples]
    return Dataset.from_dict({"text": formatted_texts})


def find_latest_gguf_export(search_dir="models"):
    gguf_files = sorted(Path(search_dir).rglob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return gguf_files[0] if gguf_files else None


def main():
    cfg = load_config()
    model_cfg = cfg["model"]

    parser = argparse.ArgumentParser(description="Fine-tune a base model with Unsloth on 8GB VRAM")
    parser.add_argument("--data", type=str, default=str(TRAIN_DATA_PATH), help="Path to training JSON")
    parser.add_argument("--max-seq-length", type=int, default=model_cfg["max_seq_length"], help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=model_cfg["epochs"], help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs with exact step count (e.g. 60)")
    parser.add_argument("--learning-rate", type=float, default=model_cfg["lr"], help="Learning rate")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run GGUF export only")
    parser.add_argument("--export-gguf", action="store_true", default=True, help="Export to GGUF after training")
    args = parser.parse_args()

    model_name = model_cfg["base_model"]
    domain_label = cfg["dataset"]["domain_label"]
    system_prompt = build_system_prompt(domain_label)
    serve_model_name = cfg["agent"]["serve_model"]

    print("=" * 60)
    print("RAFT 8GB Fine-Tuning Pipeline (Unsloth QLoRA)")
    print("=" * 60)
    print(f"Base model:         {model_name}")
    print(f"Dataset:            {args.data}")
    print(f"Max Sequence Len:   {args.max_seq_length}")
    print(f"Batch Size:         {model_cfg['batch_size']} (per device)")
    print(f"Gradient Accum:     {model_cfg['grad_accum']} (effective batch size: {model_cfg['batch_size'] * model_cfg['grad_accum']})")
    print(f"Quantization:       4-bit (load_in_4bit={model_cfg['load_in_4bit']})")
    print(f"Optimizer:          paged_adamw_8bit")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"\n[ERROR] Required fine-tuning library not found: {e}")
        print("Please install the required packages in your environment:")
        print("  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("  pip install transformers datasets trl peft bitsandbytes accelerate")
        sys.exit(1)

    print(f"\n1. Loading base model '{model_name}' in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    if not tokenizer.chat_template:
        raise RuntimeError(f"Tokenizer for model '{model_name}' has no chat template; cannot format RAFT prompts.")

    if not args.skip_train:
        print(f"\n2. Configuring LoRA target modules (r={model_cfg['lora_r']}, alpha={model_cfg['lora_alpha']})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_cfg["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=model_cfg["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=args.max_seq_length,
        )

        print("\n3. Formatting training dataset into chat template...")
        train_dataset = prepare_hf_dataset(args.data, tokenizer, system_prompt)
        print(f"   Loaded {len(train_dataset)} formatted training sequences.")

        training_args = TrainingArguments(
            per_device_train_batch_size=model_cfg["batch_size"],
            gradient_accumulation_steps=model_cfg["grad_accum"],
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        print("\n4. Starting QLoRA fine-tuning on GPU...")
        trainer_stats = trainer.train()
        print("\nTraining completed successfully!")
        print(f"Training runtime: {trainer_stats.metrics.get('train_runtime', 0):.2f}s")

        print(f"\n5. Saving LoRA adapter weights to '{LORA_OUTPUT_DIR}'...")
        LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(LORA_OUTPUT_DIR))
        tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))
        print("LoRA adapter saved successfully.")
    else:
        if not LORA_OUTPUT_DIR.exists():
            raise RuntimeError(
                f"--skip-train requires an existing fine-tuned adapter at '{LORA_OUTPUT_DIR}', but none was found."
            )
        print(f"\nSkipping training. Reloading fine-tuned adapter from '{LORA_OUTPUT_DIR}' for export...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(LORA_OUTPUT_DIR),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=model_cfg["load_in_4bit"],
        )

    if args.export_gguf:
        print(f"\n6. Exporting merged model to GGUF (Q4_K_M) at '{GGUF_OUTPUT_DIR}'...")
        GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(GGUF_OUTPUT_DIR),
            tokenizer,
            quantization_method="q4_k_m"
        )

        gguf_path = find_latest_gguf_export()
        if gguf_path is None:
            raise RuntimeError("GGUF export finished but no .gguf file was found under 'models/'.")
        modelfile_path = gguf_path.parent / "Modelfile"

        print("\nGGUF export complete! Model is ready for Ollama.")
        print(f"GGUF file:          {gguf_path}")
        if modelfile_path.exists():
            print(f"Modelfile:           {modelfile_path}")
            print(f"\nRun next:\n  ollama create {serve_model_name} -f {modelfile_path}")
        else:
            print(f"[WARN] Expected Modelfile not found at '{modelfile_path}'; check the export directory before running ollama create.")


if __name__ == "__main__":
    main()
