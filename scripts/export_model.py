# scripts/export_model.py
"""Export trained model to HuggingFace Hub.

Usage:
    python scripts/export_model.py \
        --checkpoint-dir ./checkpoints/qwen-0.5b-tool-agent \
        --repo-id yourusername/qwen-0.5b-tool-agent-grpo \
        --merge-adapter
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# 中文注释：从检查点加载 tokenizer 和模型，可选地合并 LoRA 后推送到 HuggingFace。
def export(checkpoint_dir: str, repo_id: str, commit_message: str, merge_adapter: bool):
    print(f"Loading model from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    if merge_adapter:
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()
        print("Merged LoRA adapter into base model")
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype="auto")

    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id, commit_message=commit_message)
    tokenizer.push_to_hub(repo_id, commit_message=commit_message)
    print(f"Done! Model at: https://huggingface.co/{repo_id}")


# 中文注释：命令行入口收集检查点、仓库和是否合并适配器等参数。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--commit-message", default="RL-trained tool-use agent via GRPO")
    parser.add_argument("--merge-adapter", action="store_true")
    args = parser.parse_args()
    export(args.checkpoint_dir, args.repo_id, args.commit_message, args.merge_adapter)


if __name__ == "__main__":
    main()
