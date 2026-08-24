"""LLM — the single LLM call wrapper (all model calls go through here)"""
from __future__ import annotations
from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Maximum tokens to generate (can be overridden via kw)
_DEFAULT_MAX_TOKENS = 1024


class LLM:
    """Single entry-point for all LLM calls.

    Backend priority (first that works wins):
      1. OpenAI-compatible API  (openai>=1.0; also covers Google via openai compat)
      2. Google Generative AI   (google-generativeai)
      3. HuggingFace local      (transformers pipeline, CPU-friendly)

    The API key comes from ``settings.llm_api_key`` (set in .env).
    The model name comes from ``cfg['agent']['model']`` (default: gpt-4o-mini).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        agent_cfg = cfg.get("agent", {})
        # Model name — support legacy 'model' key and shorthand names.
        self.model: str = str(agent_cfg.get("model", "gpt-4o-mini"))
        self.temperature: float = float(agent_cfg.get("temperature", 0.0))
        self.max_tokens: int = int(agent_cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))

        try:
            from ..settings import settings
            self._api_key: str = settings.llm_api_key or ""
        except Exception:
            import os
            self._api_key = os.environ.get("LLM_API_KEY", "")

    # ------------------------------------------------------------------
    def complete(self, prompt: str, **kw) -> str:
        """Call the configured LLM and return the generated text.

        Parameters
        ----------
        prompt : str
            The full prompt to send.
        **kw : dict
            Optional overrides: ``model``, ``temperature``, ``max_tokens``.

        Returns
        -------
        str
            The model's response text (stripped).
        """
        model       = kw.get("model",       self.model)
        temperature = float(kw.get("temperature", self.temperature))
        max_tokens  = int(kw.get("max_tokens",  self.max_tokens))

        logger.info(
            "LLM.complete: model=%s, prompt_chars=%d", model, len(prompt)
        )

        # ----------------------------------------------------------------
        # Backend 1 — OpenAI-compatible  (openai>=1.0)
        # ----------------------------------------------------------------
        if self._api_key:
            try:
                from openai import OpenAI  # type: ignore

                base_url: str | None = None
                # Gemini via OpenAI compat layer
                if "gemini" in model.lower():
                    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

                client = OpenAI(api_key=self._api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                logger.info("LLM: OpenAI backend succeeded, chars=%d", len(text))
                return text.strip()

            except ImportError:
                logger.warning("openai package not installed; trying next backend.")
            except Exception as exc:
                logger.warning("OpenAI backend failed (%s); trying next.", exc)

        # ----------------------------------------------------------------
        # Backend 2 — google-generativeai
        # ----------------------------------------------------------------
        if self._api_key and "gemini" in model.lower():
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=self._api_key)
                gmodel = genai.GenerativeModel(model)
                resp = gmodel.generate_content(prompt)
                text = resp.text or ""
                logger.info(
                    "LLM: google-generativeai backend succeeded, chars=%d", len(text)
                )
                return text.strip()

            except ImportError:
                logger.warning(
                    "google-generativeai not installed; trying next backend."
                )
            except Exception as exc:
                logger.warning("Google GenAI backend failed (%s); trying next.", exc)

        # ----------------------------------------------------------------
        # Backend 3 — Local agy.exe CLI fallback
        # ----------------------------------------------------------------
        import subprocess
        try:
            result = subprocess.run(
                [
                    r"C:\Users\ASUS\AppData\Local\agy\bin\agy.exe",
                    "--model",
                    "gemini-3.6-flash-high",
                    "-p",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                text = result.stdout.strip()
                logger.info("LLM: agy.exe CLI backend succeeded, chars=%d", len(text))
                return text
            else:
                logger.warning("agy.exe CLI returned no output or failed (code %d); trying next.", result.returncode)
        except FileNotFoundError:
            logger.warning("agy.exe not found at the specified path; trying next.")
        except subprocess.TimeoutExpired:
            logger.warning("agy.exe CLI timed out; trying next.")
        except Exception as exc:
            logger.warning("agy.exe CLI backend failed (%s); trying next.", exc)

        # ----------------------------------------------------------------
        # Backend 4 — HuggingFace local pipeline (CPU-friendly fallback)
        # ----------------------------------------------------------------
        try:
            from transformers import pipeline  # type: ignore
            import torch

            device = 0 if torch.cuda.is_available() else -1
            # Use a small multilingual model that can at least produce text.
            hf_model = model if "/" in model else "facebook/opt-125m"
            pipe = pipeline(
                "text-generation",
                model=hf_model,
                device=device,
                max_new_tokens=max_tokens,
            )
            output = pipe(prompt, do_sample=False, temperature=None, top_p=None)
            text = output[0]["generated_text"]
            # strip the echoed prompt
            if text.startswith(prompt):
                text = text[len(prompt):]
            logger.info("LLM: HF local backend succeeded, chars=%d", len(text))
            return text.strip()

        except ImportError:
            logger.warning("transformers not installed; no fallback available.")
        except Exception as exc:
            logger.error("HF local backend failed: %s", exc)

        raise RuntimeError(
            "LLM.complete: all backends failed. "
            "Set LLM_API_KEY in .env or install a supported backend."
        )
