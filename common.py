"""Funções e constantes compartilhadas entre treino, avaliação e app.

Concentra o formato de prompt, chat template, geração na GPU e parsing de saída.
"""

import json
import re
from datetime import datetime
from pathlib import Path

# Diretórios e caminhos padrão do projeto
if "__file__" in globals():
    PASTA = Path(__file__).resolve().parent
else:
    PASTA = Path.cwd()

BASE = "Qwen/Qwen3-1.7B"
DIR_DATA = PASTA / "data"
DIR_RESULTS = PASTA / "results"
DIR_ADAPTERS = PASTA / "adapters"
DIR_ASSETS = PASTA / "assets"

DIAS_SEMANA = [
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
]

INSTRUCAO = (
    "Converta a expressão de data/hora em português para o formato ISO 8601, "
    "usando a data de referência. Responda somente com AAAA-MM-DD; se a "
    "expressão indicar um horário, responda AAAA-MM-DDTHH:MM."
)

PADRAO_ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?")


# ---------------------------------------------------------------------------
# Formato do prompt
# ---------------------------------------------------------------------------
def prompt_chat(tok, referencia, expressao):
    """Texto final enviado ao modelo (chat template do Qwen3, sem 'thinking')."""
    if isinstance(referencia, str):
        referencia = datetime.fromisoformat(referencia)
    dia = DIAS_SEMANA[referencia.weekday()]
    conteudo_usuario = f"Referência: {dia}, {referencia:%Y-%m-%d %H:%M}\nExpressão: {expressao}"

    mensagens = [
        {"role": "system", "content": INSTRUCAO},
        {"role": "user", "content": conteudo_usuario},
    ]
    return tok.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Dispositivo e Inferência
# ---------------------------------------------------------------------------
def escolher_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gerar(model, tok, prompts, max_new_tokens=24, temperatura=0.0):
    """Gera respostas para uma lista de prompts (já no formato do chat)."""
    import torch

    entrada = tok(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    opcoes = (
        {"do_sample": True, "temperature": temperatura, "top_p": 0.95}
        if temperatura > 0
        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    )
    with torch.no_grad():
        saida = model.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            **opcoes,
        )
    novos = saida[:, entrada["input_ids"].shape[1] :]
    return [t.strip() for t in tok.batch_decode(novos, skip_special_tokens=True)]


# ---------------------------------------------------------------------------
# Extração e Parsing de Saída
# ---------------------------------------------------------------------------
def extrair_iso(texto):
    """Primeira data ISO encontrada no texto (ou None)."""
    achado = PADRAO_ISO.search(texto)
    return achado.group(0) if achado else None


# ---------------------------------------------------------------------------
# Leitura de Datasets
# ---------------------------------------------------------------------------
def ler_jsonl(nome):
    caminho = DIR_DATA / f"{nome}.jsonl"
    with open(caminho, encoding="utf-8") as arq:
        return [json.loads(l) for l in arq if l.strip()]
