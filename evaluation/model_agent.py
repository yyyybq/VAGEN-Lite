"""
VLM Model Agent for Evaluation
===============================

Wraps a Vision-Language Model (via vLLM or API) for interactive evaluation.
Handles prompt construction, image encoding, and response generation.
"""

import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional

from evaluation.agents import BaseAgent
from evaluation.eval_config import EvalModelConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ModelAgent(BaseAgent):
    """
    VLM model agent for evaluation.
    
    Supports:
    - vLLM local inference (for trained checkpoints)
    - OpenAI/Claude/Gemini API (for baseline comparison)
    """
    
    def __init__(self, model_config: EvalModelConfig):
        super().__init__()
        self.name = f"model_{model_config.model_name}"
        self.model_config = model_config
        self.model = None
        self.conversation_history: List[Dict] = []
        self.system_prompt_text: Optional[str] = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the model backend."""
        provider = self.model_config.provider
        
        if provider == "vllm":
            self._init_vllm()
        elif provider in ("openai", "claude", "gemini"):
            self._init_api_model()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _is_cambrian_checkpoint(self, model_path: str) -> bool:
        """Detect Cambrian-S HF checkpoints by config.json."""
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.is_file():
            name = (self.model_config.model_name or "").lower()
            return "cambrian" in name
        try:
            import json
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            return "cambrian" in model_path.lower()
        model_type = str(cfg.get("model_type", "")).lower()
        arches = " ".join(str(a) for a in (cfg.get("architectures") or [])).lower()
        return "cambrian" in model_type or "cambrian" in arches

    def _init_vllm(self):
        """Initialize vLLM for local inference."""
        try:
            from vllm import LLM, SamplingParams
            
            model_path = self.model_config.checkpoint_path or self.model_config.model_name
            self.is_cambrian = self._is_cambrian_checkpoint(model_path)

            print(f"Loading vLLM model: {model_path}")
            print(f"  TP size: {self.model_config.tensor_parallel_size}")
            print(f"  GPU mem: {self.model_config.gpu_memory_utilization}")
            print(f"  Cambrian: {self.is_cambrian}")

            # Ensure Cambrian architectures are registered in this process and
            # in vLLM worker subprocesses (also via vllm.general_plugins).
            if self.is_cambrian:
                try:
                    from vagen.models.cambrian_plugin import register as register_cambrian
                    register_cambrian()
                except Exception as e:
                    print(f"  Warning: cambrian_plugin.register failed: {e}")
                try:
                    import vagen.models.cambrian_vllm  # noqa: F401
                except Exception as e:
                    print(f"  Warning: import cambrian_vllm failed: {e}")

            # Cambrian + SigLIP needs eager/SDPA-friendly settings on these nodes.
            # Full 12-step eval can exceed 8 images / 8192 expanded tokens.
            # B5 ablation: override with VAGEN_CAMBRIAN_MAX_MODEL_LEN / VAGEN_CAMBRIAN_LIMIT_IMAGES.
            import os as _os
            if self.is_cambrian:
                max_model_len = int(_os.environ.get("VAGEN_CAMBRIAN_MAX_MODEL_LEN", "16384"))
                limit_images = int(_os.environ.get("VAGEN_CAMBRIAN_LIMIT_IMAGES", "25"))
            else:
                max_model_len = int(_os.environ.get("VAGEN_QWEN_MAX_MODEL_LEN", "32768"))
                limit_images = int(_os.environ.get("VAGEN_QWEN_LIMIT_IMAGES", "25"))
            enforce_eager = bool(self.is_cambrian)
            limit_mm = {"image": limit_images}

            self.model = LLM(
                model=model_path,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                gpu_memory_utilization=self.model_config.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                limit_mm_per_prompt=limit_mm,
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                max_tokens=self.model_config.max_tokens,
                stop=["</action>"],
                include_stop_str_in_output=True,
            )
            
            print(f"  Model loaded successfully.")
            
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    def _init_api_model(self):
        """Initialize API-based model (OpenAI, Claude, etc.)."""
        provider = self.model_config.provider
        
        if provider == "openai":
            try:
                from openai import OpenAI
                self.model = OpenAI(
                    api_key=self.model_config.api_key or os.environ.get("OPENAI_API_KEY"),
                    base_url=self.model_config.api_base,
                )
            except ImportError:
                raise ImportError("openai package not installed.")
        
        elif provider == "claude":
            try:
                import anthropic
                self.model = anthropic.Anthropic(
                    api_key=self.model_config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                )
            except ImportError:
                raise ImportError("anthropic package not installed.")
        
        elif provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.model_config.api_key or os.environ.get("GOOGLE_API_KEY"))
                self.model = genai.GenerativeModel(self.model_config.model_name)
            except ImportError:
                raise ImportError("google-generativeai package not installed.")
    
    def act(self, observation: Dict[str, Any], info: Dict[str, Any] = None) -> str:
        """Generate action from VLM given observation."""
        obs_str = observation.get("obs_str", "")
        multi_modal = observation.get("multi_modal_data", {})
        
        # Extract images
        images = []
        for key, values in multi_modal.items():
            if "image" in key.lower():
                images.extend(values)
        
        # Build message
        self.conversation_history.append({
            "role": "user",
            "content": obs_str,
            "images": images,
        })
        
        # Generate response
        provider = self.model_config.provider
        if provider == "vllm":
            response = self._generate_vllm(images)
        elif provider == "openai":
            response = self._generate_openai(images)
        elif provider == "claude":
            response = self._generate_claude(images)
        elif provider == "gemini":
            response = self._generate_gemini(images)
        else:
            response = "<think>No model.</think><action>move_forward|</action>"
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
        })
        
        return response
    
    def _generate_vllm(self, images: List) -> str:
        """Generate with vLLM (local model)."""
        # Cambrian expects compact <image> placeholders + multi_modal_data.
        if getattr(self, "is_cambrian", False):
            return self._generate_vllm_cambrian(images)

        # Build the full prompt for vLLM
        # For Qwen2.5-VL, we need to use the chat template
        messages = []
        
        if self.system_prompt_text:
            messages.append({"role": "system", "content": self.system_prompt_text})
        
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            msg_images = msg.get("images", [])
            
            if role == "user" and msg_images:
                # Build multimodal content
                content_parts = []
                for img in msg_images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(img)},
                    })
                content_parts.append({"type": "text", "text": content})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": role, "content": content})
        
        # Use vLLM chat
        outputs = self.model.chat(
            messages=[messages],
            sampling_params=self.sampling_params,
        )
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return "<think>Generation failed.</think><action>move_forward|</action>"

    def _generate_vllm_cambrian(self, images: List) -> str:
        """Cambrian plain-vLLM path: chat template text + multi_modal_data."""
        from transformers import AutoTokenizer

        if not hasattr(self, "_cambrian_tokenizer"):
            model_path = self.model_config.checkpoint_path or self.model_config.model_name
            self._cambrian_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        messages = []
        if self.system_prompt_text:
            messages.append({"role": "system", "content": self.system_prompt_text})

        all_images = []
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"] or ""
            msg_images = msg.get("images", []) or []
            if role == "user" and msg_images:
                # Ensure one <image> token per provided image exists in text.
                n_existing = content.count("<image>")
                if n_existing < len(msg_images):
                    prefix = "<image>" * (len(msg_images) - n_existing)
                    content = prefix + ("\n" if content else "") + content
                all_images.extend(msg_images)
            messages.append({"role": role, "content": content})

        try:
            prompt = self._cambrian_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback Qwen-style template
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(parts)

        req = {"prompt": prompt}
        if all_images:
            req["multi_modal_data"] = {"image": all_images if len(all_images) > 1 else all_images[0]}

        outputs = self.model.generate([req], sampling_params=self.sampling_params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return "<think>Generation failed.</think><action>move_forward|</action>"
    
    def _generate_openai(self, images: List) -> str:
        """Generate with OpenAI API."""
        messages = self._build_openai_messages(images)
        
        response = self.model.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
        )
        return response.choices[0].message.content
    
    def _generate_claude(self, images: List) -> str:
        """Generate with Claude API."""
        messages = self._build_claude_messages(images)
        system = self.system_prompt_text or ""
        
        response = self.model.messages.create(
            model=self.model_config.model_name,
            max_tokens=self.model_config.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text
    
    def _generate_gemini(self, images: List) -> str:
        """Generate with Gemini API."""
        # Build parts for latest message
        parts = []
        if images:
            for img in images:
                parts.append(img)  # PIL Image
        
        latest_text = self.conversation_history[-1]["content"] if self.conversation_history else ""
        parts.append(latest_text)
        
        response = self.model.generate_content(parts)
        return response.text
    
    def _build_openai_messages(self, current_images: List) -> List[Dict]:
        """Build OpenAI-format messages from conversation history."""
        messages = []
        if self.system_prompt_text:
            messages.append({"role": "system", "content": self.system_prompt_text})
        
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            msg_images = msg.get("images", [])
            
            if role == "user" and msg_images:
                content_parts = []
                for img in msg_images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(img), "detail": "low"},
                    })
                content_parts.append({"type": "text", "text": content})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": role, "content": content})
        
        return messages
    
    def _build_claude_messages(self, current_images: List) -> List[Dict]:
        """Build Claude-format messages from conversation history."""
        messages = []
        for msg in self.conversation_history:
            role = msg["role"]
            if role == "system":
                continue
            content = msg["content"]
            msg_images = msg.get("images", [])
            
            if role == "user" and msg_images:
                content_parts = []
                for img in msg_images:
                    img_data = self._image_to_base64(img)
                    content_parts.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img_data},
                    })
                content_parts.append({"type": "text", "text": content})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": role, "content": content})
        
        return messages
    
    def _image_to_base64(self, img) -> str:
        """Convert PIL Image to base64 string."""
        from PIL import Image
        if isinstance(img, Image.Image):
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        return ""
    
    def _image_to_data_url(self, img) -> str:
        """Convert PIL Image to data URL."""
        b64 = self._image_to_base64(img)
        return f"data:image/png;base64,{b64}"
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt (called by the runner after env reset)."""
        self.system_prompt_text = prompt
    
    def reset(self):
        """Reset conversation history for a new episode."""
        self.conversation_history = []
