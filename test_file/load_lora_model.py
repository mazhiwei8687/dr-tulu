from __future__ import annotations

import argparse
import os
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from default_prompt import SYSTEM_PROMPT_V20250824

HUGGINGFACE_CERT_FILE_PATH = "/Users/maz/certs/huggingface.co.crt"

os.environ["CURL_CA_BUNDLE"] = HUGGINGFACE_CERT_FILE_PATH
os.environ["REQUESTS_CA_BUNDLE"] = HUGGINGFACE_CERT_FILE_PATH
os.environ["SSL_CERT_FILE"] = HUGGINGFACE_CERT_FILE_PATH

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_LORA_PATH = "/Users/maz/downloads/qwen3-4B-sft-final"


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 Qwen3-4B LoRA 适配器并进行交互式对话"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"原始 Qwen3-4B 基座模型或本地路径 (默认: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help=f"LoRA 训练输出路径 (默认: {DEFAULT_LORA_PATH})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="单轮生成的最大新 token 数 (默认 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度 (默认 0.7)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="如模型需要自定义推理 (如 Qwen 系列) 则开启",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="手动指定推理设备；默认自动探测",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=SYSTEM_PROMPT_V20250824,
        help="可选的系统提示词，用于支持 chat 模型 (默认使用统一工具提示)",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="将 LoRA 权重并入基座模型后再推理 (占用更多显存)",
    )
    return parser.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_bundle(
    base_model_id: str,
    lora_path: str,
    device: str,
    trust_remote_code: bool,
    merge_lora: bool,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    if merge_lora:
        model = model.merge_and_unload()
        if device == "cuda":
            model.to("cuda")
    elif device != "cuda":
        model = model.to(device)
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model, device=device)


def build_inputs(tokenizer: AutoTokenizer, user_prompt: str, system_prompt: Optional[str]) -> dict:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        content = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(content, return_tensors="pt")
    return tokenizer(user_prompt, return_tensors="pt")


def generate_reply(
    bundle: ModelBundle,
    prompt: str,
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = build_inputs(bundle.tokenizer, prompt, system_prompt)
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    generated = output_ids[0]
    if generated.shape[0] <= inputs["input_ids"].shape[1]:
        return "(未生成输出)"
    new_tokens = generated[inputs["input_ids"].shape[1]:]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_chat(
    bundle: ModelBundle,
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
) -> None:
    print("进入 LoRA 模型交互模式，按 Ctrl+C 或输入 /exit 退出。\n")
    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit", ":q"}:
            print("再见！")
            break
        reply = generate_reply(bundle, user, system_prompt, max_new_tokens, temperature)
        print(f"模型: {reply}\n")


def setup_signal_handlers() -> None:
    def handle_sigint(signum, frame):  # noqa: ANN001
        print("\n检测到中断，安全退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)


def main() -> None:
    args = parse_args()
    setup_signal_handlers()

    device = pick_device(args.device)
    print(f"正在 {device} 上加载基座 {args.base_model} 与 LoRA {args.lora_path} …")
    bundle = load_bundle(
        base_model_id=args.base_model,
        lora_path=args.lora_path,
        device=device,
        trust_remote_code=args.trust_remote_code,
        merge_lora=args.merge_lora,
    )
    print("加载完成，可以开始对话。\n")

    interactive_chat(
        bundle,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
