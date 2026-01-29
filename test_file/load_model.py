from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

from default_prompt import SYSTEM_PROMPT_V20250824

HUGGINGFACE_CERT_FILE_PATH = "/Users/maz/certs/huggingface.co.crt"

os.environ["CURL_CA_BUNDLE"] = HUGGINGFACE_CERT_FILE_PATH
os.environ["REQUESTS_CA_BUNDLE"] = HUGGINGFACE_CERT_FILE_PATH
os.environ["SSL_CERT_FILE"] = HUGGINGFACE_CERT_FILE_PATH

# DEFAULT_MODEL_PATH = "rl-research/DR-Tulu-8B"
# DEFAULT_MODEL_PATH = "rl-research/DR-Tulu-SFT-8B"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
# DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B"
# DEFAULT_MODEL_PATH = "Qwen/Qwen3-0.6B"


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 HuggingFace 上的 Causal LM，并在终端与其交互"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"HuggingFace 上的模型 ID 或本地路径 (默认: {DEFAULT_MODEL_PATH})",
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
        help="可选的系统提示词，用于支持 chat 模型",
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


def load_bundle(model_id: str, device: str, trust_remote_code: bool) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
    )
    if device != "cuda":
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


def generate_reply(bundle: ModelBundle, prompt: str, system_prompt: Optional[str], max_new_tokens: int,
                   temperature: float) -> str:
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


def interactive_chat(bundle: ModelBundle, system_prompt: Optional[str], max_new_tokens: int,
                     temperature: float) -> None:
    print("进入交互模式，按 Ctrl+C 或输入 /exit 退出。\n")

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
    print(f"正在 {device} 上加载 {args.model} …")
    bundle = load_bundle(args.model, device, args.trust_remote_code)
    print("加载完成，可以开始对话。\n")

    interactive_chat(
        bundle,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
