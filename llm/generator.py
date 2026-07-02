"""Geração da resposta final com um LLM local (Hugging Face Transformers).

Após a busca semântica recuperar as sinopses mais relevantes, um LLM local
e leve (ex.: SmolLM, TinyLlama, Phi-3 Mini, Mistral) formata uma resposta
em linguagem natural no estilo RAG (Retrieval-Augmented Generation).

O modelo padrão é o ``SmolLM2-360M-Instruct``, um modelo instruct leve que
roda em CPU e usa o *chat template* do tokenizer.
"""

from __future__ import annotations

from typing import Any

from tf import config

# Template de prompt para o padrão RAG. O contexto é preenchido com as
# sinopses recuperadas e a pergunta original do usuário.
PROMPT_TEMPLATE = """Você é um assistente especializado em filmes. Use apenas as
sinopses fornecidas como contexto para responder à pergunta do usuário.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

# Número máximo de caracteres de cada sinopse incluída no contexto, para
# evitar estourar a janela de contexto do modelo.
_MAX_PLOT_CHARS = 800


class LocalLLM:
    """Encapsula o carregamento e a inferência de um LLM local."""

    def __init__(
        self,
        model_name: str = config.LLM_MODEL_NAME,
        max_new_tokens: int = config.LLM_MAX_NEW_TOKENS,
    ) -> None:
        """Configura o LLM.

        Args:
            model_name: Nome do modelo no Hugging Face.
            max_new_tokens: Limite de tokens gerados na resposta.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> "LocalLLM":
        """Carrega o tokenizer e o modelo (Transformers) na memória/dispositivo.

        Returns:
            A própria instância.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
        ).to(device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self

    def _ensure_loaded(self) -> None:
        """Carrega o modelo sob demanda, caso ainda não tenha sido carregado."""
        if self._model is None or self._tokenizer is None:
            self.load()

    @staticmethod
    def _format_context(retrieved_docs: list[dict[str, Any]]) -> str:
        """Formata as sinopses recuperadas em um bloco de contexto legível."""
        blocks: list[str] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            title = doc.get("title") or "Título desconhecido"
            plot = (doc.get("plot") or "").strip()
            if len(plot) > _MAX_PLOT_CHARS:
                plot = plot[:_MAX_PLOT_CHARS].rstrip() + "..."
            blocks.append(f"[{i}] {title}\n{plot}")
        return "\n\n".join(blocks) if blocks else "(nenhuma sinopse encontrada)"

    def build_prompt(self, question: str, retrieved_docs: list[dict[str, Any]]) -> str:
        """Monta o prompt RAG a partir da pergunta e das sinopses recuperadas.

        Args:
            question: Pergunta em linguagem natural do usuário.
            retrieved_docs: Documentos recuperados pela busca (com título/sinopse).

        Returns:
            Prompt final pronto para o modelo.
        """
        context = self._format_context(retrieved_docs)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate_answer(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        """Gera a resposta em linguagem natural.

        Args:
            question: Pergunta do usuário.
            retrieved_docs: Sinopses recuperadas pela busca semântica.

        Returns:
            Texto da resposta formatada.

        Complexidade: O(L) no número de tokens gerados (custo de inferência).
        """
        import torch

        self._ensure_loaded()

        prompt = self.build_prompt(question, retrieved_docs)

        # Usa o chat template do modelo instruct quando disponível, obtendo o
        # texto formatado e tokenizando em seguida (comportamento estável entre
        # versões do Transformers).
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            text = prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decodifica apenas os tokens novos (após o prompt).
        generated = output_ids[0, prompt_len:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True)
        return answer.strip()
