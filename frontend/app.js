/**
 * ISO 8601 Studio — Client Controller
 */

const DIAS_SEMANA = [
  "domingo",
  "segunda-feira",
  "terça-feira",
  "quarta-feira",
  "quinta-feira",
  "sexta-feira",
  "sábado"
];

// Elementos da Interface
const inputRef = document.getElementById("inputRef");
const inputExpr = document.getElementById("inputExpr");
const btnAgora = document.getElementById("btnAgora");
const btnConverter = document.getElementById("btnConverter");
const btnSpinner = btnConverter.querySelector(".btn-spinner");
const helperRefDia = document.getElementById("helperRefDia");

// Seletor de Adaptador LoRA
const modelSelector = document.getElementById("modelSelector");
let modeloLoraAtivo = "r16";

// Painel Esquerdo (Qwen Base — Referência)
const outputBase = document.getElementById("outputBase");
const latencyBase = document.getElementById("latencyBase");
const statusTextBase = document.getElementById("statusTextBase");
const statusBulletBase = document.querySelector("#statusBase .status-bullet");
const btnCopyBase = document.getElementById("btnCopyBase");

// Painel Direito (Modelo LoRA Especializado)
const titleLora = document.getElementById("titleLora");
const outputLora = document.getElementById("outputLora");
const latencyLora = document.getElementById("latencyLora");
const statusTextLora = document.getElementById("statusTextLora");
const statusBulletLora = document.querySelector("#statusLora .status-bullet");
const captionLora = document.getElementById("captionLora");
const btnCopyLora = document.getElementById("btnCopyLora");

const presetsList = document.getElementById("presetsList");
const gpuStatusText = document.getElementById("gpuStatusText");

// Checagem de status da GPU / Hardware
async function checarStatusBackend() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      const hw = data.hardware || "Online";
      const vram = data.vram ? ` • ${data.vram}` : "";
      gpuStatusText.textContent = `${hw}${vram} (${data.device})`;
    } else {
      gpuStatusText.textContent = "Online";
    }
  } catch (err) {
    gpuStatusText.textContent = "Local";
  }
}
checarStatusBackend();

// Atualiza o dia da semana no helper ao alterar a data
function atualizarDiaSemana() {
  try {
    const val = inputRef.value.trim().replace(" ", "T");
    const d = new Date(val);
    if (!isNaN(d.getTime())) {
      const idx = d.getDay();
      helperRefDia.textContent = `(${DIAS_SEMANA[idx] || ""})`;
      return;
    }
  } catch (e) {}
  helperRefDia.textContent = "";
}

inputRef.addEventListener("input", atualizarDiaSemana);
atualizarDiaSemana();

// Botão Agora
btnAgora.addEventListener("click", () => {
  const agora = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const str = `${agora.getFullYear()}-${pad(agora.getMonth() + 1)}-${pad(agora.getDate())} ${pad(agora.getHours())}:${pad(agora.getMinutes())}`;
  inputRef.value = str;
  atualizarDiaSemana();
});

// Seletor de Modelo LoRA (r=16 vs r=8)
modelSelector.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    modelSelector.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    modeloLoraAtivo = btn.dataset.model;
    atualizarCardLoraInfo();
    if (inputExpr.value.trim()) {
      executarConversao();
    }
  });
});

function atualizarCardLoraInfo() {
  if (modeloLoraAtivo === "r16") {
    titleLora.textContent = "Modelo LoRA Especializado (r=16)";
    captionLora.textContent = "Adaptador principal (17.4M parâmetros) com ajuste de atenção e MLP.";
  } else {
    titleLora.textContent = "Modelo LoRA Compacto (r=8)";
    captionLora.textContent = "Adaptador compacto (8.7M parâmetros) com 0,5% de parâmetros treináveis.";
  }
}
atualizarCardLoraInfo();

// Presets Rápidos
presetsList.querySelectorAll(".preset-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    inputRef.value = chip.dataset.ref;
    inputExpr.value = chip.dataset.expr;
    atualizarDiaSemana();
    executarConversao();
  });
});

// Tecla Enter no input de expressão
inputExpr.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    executarConversao();
  }
});

// Botão Converter
btnConverter.addEventListener("click", executarConversao);

// Função Principal de Conversão
async function executarConversao() {
  const ref = inputRef.value.trim();
  const expr = inputExpr.value.trim();

  if (!expr) {
    inputExpr.focus();
    return;
  }

  btnConverter.disabled = true;
  btnSpinner.classList.remove("hidden");
  outputBase.style.opacity = "0.5";
  outputLora.style.opacity = "0.5";

  try {
    const resp = await fetch("/api/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        referencia: ref,
        expressao: expr,
        modelo_lora: modeloLoraAtivo
      })
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `Erro na API: ${resp.status}`);
    }

    const data = await resp.json();
    const resBase = data.base;
    const resLora = data.lora;

    // 1. Atualizar Painel Esquerdo (Qwen Base — Referência)
    outputBase.textContent = resBase.resposta || "—";
    outputBase.classList.remove("output-empty");
    latencyBase.textContent = `${Math.round(resBase.latencia_ms)} ms`;

    // 2. Atualizar Painel Direito (Modelo LoRA Especializado)
    outputLora.textContent = resLora.resposta || "—";
    outputLora.classList.remove("output-empty");
    latencyLora.textContent = `${Math.round(resLora.latencia_ms)} ms`;

    // Verificação de concordância objetiva entre os dois modelos
    const modelosConcordam = (resBase.resposta === resLora.resposta) && resBase.eh_estrito && resLora.eh_estrito;

    if (modelosConcordam) {
      // Ambos concordam na mesma data estrita
      statusTextBase.textContent = "Formato ISO 8601 válido (concorda com LoRA)";
      statusBulletBase.className = "status-bullet bullet-success";
      outputBase.classList.remove("output-raw");

      statusTextLora.textContent = "Formato ISO 8601 válido (concorda com Base)";
      statusBulletLora.className = "status-bullet bullet-success";
      outputLora.classList.remove("output-raw");
    } else {
      // Há divergência entre os modelos
      // Avaliação sintática do Painel Base:
      if (resBase.eh_estrito) {
        statusTextBase.textContent = "Divergência de cálculo";
        statusBulletBase.className = "status-bullet bullet-amber";
        outputBase.classList.remove("output-raw"); // O formato ISO é válido, sem texto vermelho de erro
      } else if (resBase.iso) {
        statusTextBase.textContent = `Resposta com texto extra (ISO: ${resBase.iso})`;
        statusBulletBase.className = "status-bullet bullet-amber";
        outputBase.classList.add("output-raw");
      } else {
        statusTextBase.textContent = "Formato inválido (nenhuma data ISO detectada)";
        statusBulletBase.className = "status-bullet bullet-red";
        outputBase.classList.add("output-raw");
      }

      // Avaliação sintática do Painel LoRA:
      if (resLora.eh_estrito) {
        statusTextLora.textContent = "Formato ISO 8601 válido (diverge do Base)";
        statusBulletLora.className = "status-bullet bullet-success";
        outputLora.classList.remove("output-raw");
      } else if (resLora.iso) {
        statusTextLora.textContent = `Resposta com texto extra (ISO: ${resLora.iso})`;
        statusBulletLora.className = "status-bullet bullet-amber";
        outputLora.classList.add("output-raw");
      } else {
        statusTextLora.textContent = "Formato inválido (nenhuma data ISO detectada)";
        statusBulletLora.className = "status-bullet bullet-red";
        outputLora.classList.add("output-raw");
      }
    }

  } catch (err) {
    console.error("Erro na conversão:", err);
    outputLora.textContent = "Erro de conexão";
    statusTextLora.textContent = String(err.message || err);
  } finally {
    btnConverter.disabled = false;
    btnSpinner.classList.add("hidden");
    outputBase.style.opacity = "1";
    outputLora.style.opacity = "1";
  }
}

// Copiar Saída Base
btnCopyBase.addEventListener("click", () => {
  const texto = outputBase.textContent.trim();
  if (texto && texto !== "—") {
    navigator.clipboard.writeText(texto).then(() => {
      const orig = btnCopyBase.innerHTML;
      btnCopyBase.innerHTML = `<span style="font-size:10px; color:#3fb950; font-weight:600;">OK</span>`;
      setTimeout(() => { btnCopyBase.innerHTML = orig; }, 1200);
    });
  }
});

// Copiar Saída LoRA
btnCopyLora.addEventListener("click", () => {
  const texto = outputLora.textContent.trim();
  if (texto && texto !== "—") {
    navigator.clipboard.writeText(texto).then(() => {
      const orig = btnCopyLora.innerHTML;
      btnCopyLora.innerHTML = `<span style="font-size:10px; color:#3fb950; font-weight:600;">OK</span>`;
      setTimeout(() => { btnCopyLora.innerHTML = orig; }, 1200);
    });
  }
});
