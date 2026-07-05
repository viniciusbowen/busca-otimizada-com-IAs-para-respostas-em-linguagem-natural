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
Responda de forma clara e objetiva, no mesmo idioma da pergunta.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

# Template para reescrever a pergunta como uma consulta de busca em inglês.
# O corpus (sinopses do CMU) está em inglês, e tanto os embeddings quanto a
# busca vetorial recuperam melhor quando a consulta também está em inglês.
QUERY_REWRITE_TEMPLATE = """You optimize search queries for a movie plot search engine.
Rewrite the user's question as a short English search query containing the key
entities, themes and keywords (no full sentences). Translate to English if needed.
Return ONLY the query text, with no quotes, labels or explanations.

User question: {question}
Search query:"""

# Número máximo de caracteres de cada sinopse incluída no contexto, para
# evitar estourar a janela de contexto do modelo.
_MAX_PLOT_CHARS = 800

# Limite de tokens gerados na etapa de reescrita da consulta (curta por natureza).
_QUERY_MAX_NEW_TOKENS = 1024


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

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Gera texto a partir de um prompt (etapa genérica de inferência).

        Args:
            prompt: Texto do prompt já montado.
            max_new_tokens: Limite de tokens gerados.

        Returns:
            Somente o texto gerado (sem o prompt), já sem espaços nas pontas.

        Complexidade: O(L) no número de tokens gerados (custo de inferência).
        """
        import torch

        self._ensure_loaded()

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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decodifica apenas os tokens novos (após o prompt).
        generated = output_ids[0, prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def rewrite_query(self, question: str) -> str:
        """Reescreve a pergunta como uma consulta de busca curta em inglês.

        Etapa opcional antes da busca: como o corpus e os embeddings funcionam
        melhor em inglês, o LLM traduz/condensa a pergunta em palavras-chave.
        Em caso de falha ou saída vazia, retorna a pergunta original.

        Args:
            question: Pergunta em linguagem natural (qualquer idioma).

        Returns:
            Consulta de busca (idealmente em inglês, só com palavras-chave).
        """
        prompt = QUERY_REWRITE_TEMPLATE.format(question=question.strip())
        raw = self._generate(prompt, max_new_tokens=_QUERY_MAX_NEW_TOKENS)

        # Pós-processa: primeira linha não vazia, sem rótulos/aspas comuns.
        query = ""
        for line in raw.splitlines():
            line = line.strip()
            # Remove rótulos que o modelo às vezes repete no início.
            for prefix in ("Search query:", "Query:", "query:"):
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix):].strip()
            # Só então remove aspas envolventes.
            line = line.strip('"').strip("'").strip()
            if line:
                query = line
                break

        return query or question.strip()

    def generate_answer(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        """Gera (formata) a resposta final em linguagem natural.

        Args:
            question: Pergunta do usuário.
            retrieved_docs: Sinopses recuperadas pela busca semântica.

        Returns:
            Texto da resposta formatada.
        """
        prompt = self.build_prompt(question, retrieved_docs)
        return self._generate(prompt, max_new_tokens=self.max_new_tokens)
