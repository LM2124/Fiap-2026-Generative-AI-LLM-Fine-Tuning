# ==============================================================================
# Servidor Backend FastAPI — ISO 8601 Studio (Qwen3-1.7B LoRA Inference)
# ==============================================================================

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

from common import (
    BASE,
    DIR_ADAPTERS,
    escolher_device,
    extrair_iso,
    gerar,
    prompt_chat,
)

PASTA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DIR_FRONTEND = PASTA / "frontend"

DEVICE = escolher_device()
modelo_peft = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo_peft, tokenizer
    print(f"\n[Backend] Inicializando Qwen3-1.7B no dispositivo: {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.padding_side = "left"

    if DEVICE == "cuda":
        dtype = torch.bfloat16
    elif DEVICE == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_base = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).to(DEVICE)

    caminho_r16 = DIR_ADAPTERS / "lora_r16"
    caminho_r8 = DIR_ADAPTERS / "lora_r8"

    if caminho_r16.exists():
        modelo_peft = PeftModel.from_pretrained(model_base, str(caminho_r16), adapter_name="r16")
    elif caminho_r8.exists():
        modelo_peft = PeftModel.from_pretrained(model_base, str(caminho_r8), adapter_name="r8")

    if caminho_r8.exists() and "r8" not in modelo_peft.peft_config:
        modelo_peft.load_adapter(str(caminho_r8), adapter_name="r8")

    modelo_peft.eval()
    print("[Backend] Adaptadores LoRA carregados:", list(modelo_peft.peft_config.keys()))
    yield
    print("[Backend] Desligando servidor...")


app = FastAPI(title="ISO 8601 Studio API", lifespan=lifespan)


class ConvertRequest(BaseModel):
    referencia: str
    expressao: str
    modelo_lora: str = "r16"


def executar_predicao(chave_modelo: str, ref_dt: datetime, expressao: str):
    prompt = prompt_chat(tokenizer, ref_dt, expressao)
    t0 = time.time()
    if chave_modelo == "base":
        with modelo_peft.disable_adapter():
            resp = gerar(modelo_peft, tokenizer, [prompt], max_new_tokens=24)[0]
    else:
        if chave_modelo not in modelo_peft.peft_config:
            chave_modelo = list(modelo_peft.peft_config.keys())[0]
        modelo_peft.set_adapter(chave_modelo)
        resp = gerar(modelo_peft, tokenizer, [prompt], max_new_tokens=24)[0]
    latencia_ms = (time.time() - t0) * 1000

    iso = extrair_iso(resp)
    eh_estrito = (iso == resp) and (iso is not None)
    return {
        "modelo": chave_modelo,
        "resposta": resp,
        "iso": iso,
        "eh_estrito": eh_estrito,
        "latencia_ms": latencia_ms,
    }, prompt


@app.post("/api/convert")
def api_convert(req: ConvertRequest):
    if not req.expressao.strip():
        raise HTTPException(status_code=400, detail="Expressão temporal não pode ser vazia.")

    try:
        ref_dt = datetime.fromisoformat(req.referencia.strip().replace(" ", "T"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data de referência inválida: {e}")

    chave_lora = req.modelo_lora if req.modelo_lora in ("r16", "r8") else "r16"

    res_base, prompt_formatado = executar_predicao("base", ref_dt, req.expressao.strip())
    res_lora, _ = executar_predicao(chave_lora, ref_dt, req.expressao.strip())

    return {
        "sucesso": True,
        "referencia": req.referencia,
        "expressao": req.expressao,
        "modelo_lora": chave_lora,
        "base": res_base,
        "lora": res_lora,
        "prompt_chatml": prompt_formatado,
    }


@app.get("/api/status")
def api_status():
    vram_str = None
    if torch.cuda.is_available():
        hw = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        vram_str = f"{vram_gb:.1f} GB" if (vram_gb % 1 >= 0.05) else f"{int(round(vram_gb))} GB"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        hw = "Apple Silicon (MPS)"
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            vram_str = f"{round(ram_gb)} GB Unificada"
        except Exception:
            vram_str = "Memória Unificada"
    else:
        hw = "CPU"
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            vram_str = f"{round(ram_gb)} GB RAM"
        except Exception:
            vram_str = None

    return {
        "status": "online",
        "device": DEVICE,
        "hardware": hw,
        "vram": vram_str,
    }


@app.get("/")
def serve_root():
    idx = DIR_FRONTEND / "index.html"
    if idx.exists():
        return FileResponse(str(idx), media_type="text/html; charset=utf-8")
    raise HTTPException(status_code=404, detail="Frontend index.html não encontrado.")


@app.get("/{filename:path}")
def serve_static(filename: str):
    arquivo = DIR_FRONTEND / filename
    if arquivo.exists() and arquivo.is_file():
        media_type = "text/css" if filename.endswith(".css") else ("application/javascript" if filename.endswith(".js") else None)
        return FileResponse(str(arquivo), media_type=media_type)
    return FileResponse(str(DIR_FRONTEND / "index.html"))


if __name__ == "__main__":
    porta = 7860
    print(f"\n========================================================")
    print(f"  ISO 8601 STUDIO — SERVIDOR FASTAPI")
    print(f"  Acesse no navegador: http://127.0.0.1:{porta}")
    print(f"========================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=porta)
