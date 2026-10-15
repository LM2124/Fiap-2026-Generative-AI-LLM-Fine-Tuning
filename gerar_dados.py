"""Geração procedural do dataset de datas em linguagem natural (PT-BR) para ISO 8601.

Permite gerar os conjuntos de treino, validação e teste com gabarito exato (datetime)
e suporta customização do tamanho dos dados via CLI ou chamada de função.
"""
import argparse
import calendar
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from common import DIR_DATA

# ---------------------------------------------------------------------------
# Vocabulário e Utilitários de Texto
# ---------------------------------------------------------------------------
EXTENSO = {
    1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco", 6: "seis",
    7: "sete", 8: "oito", 9: "nove", 10: "dez", 11: "onze", 12: "doze",
    15: "quinze", 20: "vinte", 30: "trinta",
}
DIAS_CURTOS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
MESES = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]


def numero_extenso(n: int, feminino: bool = False) -> str:
    """Retorna o número em algarismo ou por extenso alternadamente."""
    if n not in EXTENSO or random.random() < 0.5:
        return str(n)
    if feminino and n in (1, 2):
        return {1: "uma", 2: "duas"}[n]
    return EXTENSO[n]


def formatar_quantidade(n: int, singular: str, plural: str, feminino: bool = False) -> str:
    return f"{numero_extenso(n, feminino)} {singular if n == 1 else plural}"


def nome_dia_semana(wd: int) -> str:
    curto = DIAS_CURTOS[wd]
    return curto if wd >= 5 or random.random() < 0.5 else f"{curto}-feira"


def eh_masculino(wd: int) -> bool:
    return wd >= 5  # sábado e domingo são masculinos


def sortear_hora() -> tuple[int, int]:
    h = random.randint(6, 23)
    m = random.choice([0, 0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 50, 5])
    return h, m


def texto_hora(h: int, m: int) -> str:
    opcoes = [f"às {h}h" if m == 0 else f"às {h}h{m:02d}", f"às {h}:{m:02d}"]
    if h == 12 and m in (0, 30):
        opcoes.append("ao meio-dia" + (" e meia" if m else ""))
    periodo = "manhã" if 5 <= h <= 11 else "tarde" if 13 <= h <= 18 else "noite" if 19 <= h <= 23 else None
    if periodo and m in (0, 30):
        h12 = h if h < 12 else h - 12
        artigo = "à" if h12 == 1 else "às"
        opcoes.append(f"{artigo} {numero_extenso(h12, feminino=True)}{' e meia' if m else ''} da {periodo}")
    return random.choice(opcoes)


def anexar_horario(expr: str, d: date) -> tuple[str, datetime]:
    h, m = sortear_hora()
    return f"{expr} {texto_hora(h, m)}", datetime(d.year, d.month, d.day, h, m)


# ---------------------------------------------------------------------------
# Regras Aritméticas de Calendário
# ---------------------------------------------------------------------------
def somar_meses(d: date, n: int) -> date:
    total = d.month - 1 + n
    ano, mes = d.year + total // 12, total % 12 + 1
    return d.replace(year=ano, month=mes, day=min(d.day, calendar.monthrange(ano, mes)[1]))


def ultimo_dia_mes(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def proximo_dia_semana(d: date, wd: int) -> date:
    return d + timedelta(days=(wd - d.weekday()) % 7 or 7)


def passado_dia_semana(d: date, wd: int) -> date:
    return d - timedelta(days=(d.weekday() - wd) % 7 or 7)


def na_semana_que_vem(d: date, wd: int) -> date:
    return d - timedelta(days=d.weekday()) + timedelta(weeks=1, days=wd)


def proximo_dia_do_mes(d: date, dia: int) -> date:
    ano, mes = d.year, d.month
    while dia > calendar.monthrange(ano, mes)[1] or date(ano, mes, dia) < d:
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return date(ano, mes, dia)


def proxima_data_anual(d: date, dia: int, mes: int) -> date:
    alvo = date(d.year, mes, dia)
    return alvo if alvo >= d else date(d.year + 1, mes, dia)


def formatar_iso(alvo: date | datetime) -> str:
    return f"{alvo:%Y-%m-%dT%H:%M}" if isinstance(alvo, datetime) else f"{alvo:%Y-%m-%d}"


# ---------------------------------------------------------------------------
# Famílias de Expressões Temporais
# ---------------------------------------------------------------------------
def _gen_dia_relativo(ref: datetime):
    expr, delta = random.choice([
        ("hoje", 0), ("amanhã", 1), ("depois de amanhã", 2),
        ("ontem", -1), ("anteontem", -2)
    ])
    return expr, ref.date() + timedelta(days=delta)


def _gen_daqui_dias(ref: datetime):
    n = random.randint(1, 45)
    q = formatar_quantidade(n, "dia", "dias")
    return random.choice([f"daqui a {q}", f"em {q}", f"dentro de {q}"]), ref.date() + timedelta(days=n)


def _gen_dias_atras(ref: datetime):
    n = random.randint(1, 30)
    q = formatar_quantidade(n, "dia", "dias")
    return random.choice([f"há {q}", f"{q} atrás"]), ref.date() - timedelta(days=n)


def _gen_daqui_semanas(ref: datetime):
    n = random.randint(1, 8)
    q = formatar_quantidade(n, "semana", "semanas", feminino=True)
    return random.choice([f"daqui a {q}", f"em {q}", f"dentro de {q}"]), ref.date() + timedelta(weeks=n)


def _gen_proximo_dia_semana(ref: datetime):
    wd = random.randrange(7)
    nome = nome_dia_semana(wd)
    prox = "próximo" if eh_masculino(wd) else "próxima"
    expr = random.choice([
        f"{prox} {nome}",
        f"na {prox} {nome}" if not eh_masculino(wd) else f"no {prox} {nome}",
        f"{nome} que vem"
    ])
    return expr, proximo_dia_semana(ref.date(), wd)


def _gen_dia_semana_passado(ref: datetime):
    wd = random.randrange(7)
    nome = nome_dia_semana(wd)
    if eh_masculino(wd):
        expr = random.choice([f"{nome} passado", f"último {nome}", f"no {nome} passado"])
    else:
        expr = random.choice([f"{nome} passada", f"última {nome}", f"na {nome} passada"])
    return expr, passado_dia_semana(ref.date(), wd)


def _gen_semana_que_vem(ref: datetime):
    wd = random.randrange(7)
    nome = nome_dia_semana(wd)
    artigo = "no" if eh_masculino(wd) else "na"
    expr = random.choice([
        f"{nome} da semana que vem",
        f"{nome} da próxima semana",
        f"na semana que vem, {artigo} {nome}"
    ])
    return expr, na_semana_que_vem(ref.date(), wd)


def _gen_dia_do_mes(ref: datetime):
    dia = random.randint(1, 31)
    return random.choice([f"dia {dia}", f"no dia {dia}"]), proximo_dia_do_mes(ref.date(), dia)


def _gen_data_explicita(ref: datetime):
    mes = random.randint(1, 12)
    dia = random.randint(1, 28 if mes == 2 else calendar.monthrange(2025, mes)[1])
    if random.random() < 0.3:
        ano = ref.year + random.choice([0, 1, 1, 2])
        expr = random.choice([f"{dia} de {MESES[mes - 1]} de {ano}", f"{dia:02d}/{mes:02d}/{ano}"])
        return expr, date(ano, mes, dia)
    expr = random.choice([f"{dia} de {MESES[mes - 1]}", f"em {dia} de {MESES[mes - 1]}", f"{dia:02d}/{mes:02d}"])
    return expr, proxima_data_anual(ref.date(), dia, mes)


def _gen_inicio_fim_mes(ref: datetime):
    d = ref.date()
    mes_que_vem = somar_meses(d.replace(day=1), 1)
    textos, alvo = random.choice([
        (["fim do mês", "último dia do mês"], ultimo_dia_mes(d)),
        (["início do mês que vem", "primeiro dia do mês que vem", "primeiro dia do próximo mês"], mes_que_vem),
        (["último dia do mês que vem", "fim do mês que vem"], ultimo_dia_mes(mes_que_vem)),
    ])
    return random.choice(textos), alvo


def _gen_daqui_meses(ref: datetime):
    n = random.randint(1, 12)
    q = formatar_quantidade(n, "mês", "meses")
    return random.choice([f"daqui a {q}", f"em {q}", f"dentro de {q}"]), somar_meses(ref.date(), n)


def _gen_daqui_horas(ref: datetime):
    n = random.randint(1, 12)
    q = formatar_quantidade(n, "hora", "horas", feminino=True)
    return random.choice([f"daqui a {q}", f"em {q}", f"dentro de {q}"]), ref + timedelta(hours=n)


def _gen_daqui_minutos(ref: datetime):
    modelo, minutos = random.choice([
        ("daqui a {q}", None), ("em {q}", None), ("em meia hora", 30),
        ("daqui a meia hora", 30), ("daqui a uma hora e meia", 90),
    ])
    if minutos is None:
        minutos = random.choice([5, 10, 15, 20, 30, 40, 45, 50, 90, 120])
        q = formatar_quantidade(minutos, "minuto", "minutos")
        modelo = modelo.format(q=q)
    return modelo, ref + timedelta(minutes=minutos)


# Famílias Inéditas (Generalização)
def _gen_semanas_atras(ref: datetime):
    n = random.randint(1, 8)
    q = formatar_quantidade(n, "semana", "semanas", feminino=True)
    return random.choice([f"há {q}", f"{q} atrás"]), ref.date() - timedelta(weeks=n)


def _gen_meses_atras(ref: datetime):
    n = random.randint(1, 12)
    q = formatar_quantidade(n, "mês", "meses")
    return random.choice([f"há {q}", f"{q} atrás"]), somar_meses(ref.date(), -n)


def _gen_daqui_anos(ref: datetime):
    n = random.randint(1, 5)
    q = formatar_quantidade(n, "ano", "anos")
    return random.choice([f"daqui a {q}", f"em {q}", f"dentro de {q}"]), somar_meses(ref.date(), 12 * n)


def _gen_horas_atras(ref: datetime):
    n = random.randint(1, 12)
    q = formatar_quantidade(n, "hora", "horas", feminino=True)
    return random.choice([f"há {q}", f"{q} atrás"]), ref - timedelta(hours=n)


# Mapeamento: chave -> (função_geradora, combinável_com_hora, inedita)
FAMILIAS = {
    # Conhecidas no Treino (13 famílias)
    "dia_relativo":       (_gen_dia_relativo, True, False),
    "daqui_dias":         (_gen_daqui_dias, True, False),
    "dias_atras":         (_gen_dias_atras, False, False),
    "daqui_semanas":      (_gen_daqui_semanas, False, False),
    "proximo_dia_semana": (_gen_proximo_dia_semana, True, False),
    "dia_semana_passado": (_gen_dia_semana_passado, True, False),
    "semana_que_vem":     (_gen_semana_que_vem, True, False),
    "dia_do_mes":         (_gen_dia_do_mes, True, False),
    "data_explicita":     (_gen_data_explicita, True, False),
    "inicio_fim_mes":     (_gen_inicio_fim_mes, False, False),
    "daqui_meses":        (_gen_daqui_meses, False, False),
    "daqui_horas":        (_gen_daqui_horas, False, False),
    "daqui_minutos":      (_gen_daqui_minutos, False, False),
    # Inéditas no Treino (4 famílias)
    "semanas_atras":      (_gen_semanas_atras, False, True),
    "meses_atras":        (_gen_meses_atras, False, True),
    "daqui_anos":         (_gen_daqui_anos, False, True),
    "horas_atras":        (_gen_horas_atras, False, True),
}

FAMILIAS_TREINO = [f for f, (_, _, inedita) in FAMILIAS.items() if not inedita]
FAMILIAS_INEDITAS = [f for f, (_, _, inedita) in FAMILIAS.items() if inedita]


# ---------------------------------------------------------------------------
# Motor de Amostragem e Geração dos Datasets
# ---------------------------------------------------------------------------
def sortear_data_referencia(inicio: date = date(2024, 1, 1), fim: date = date(2028, 12, 31)) -> datetime:
    delta_dias = (fim - inicio).days
    d = inicio + timedelta(days=random.randrange(delta_dias + 1))
    return datetime(d.year, d.month, d.day, random.randint(7, 22), random.randrange(0, 60, 5))


def gerar_exemplo(familia: str, vistos: set | None = None) -> dict:
    fn, combinavel, _ = FAMILIAS[familia]
    while True:
        ref = sortear_data_referencia()
        expr, alvo = fn(ref)
        if combinavel and random.random() < 0.5:
            alvo = alvo.date() if isinstance(alvo, datetime) else alvo
            expr, alvo = anexar_horario(expr, alvo)
        chave = (ref, expr)
        if vistos is None or chave not in vistos:
            if vistos is not None:
                vistos.add(chave)
            return {
                "familia": familia,
                "referencia": f"{ref:%Y-%m-%dT%H:%M}",
                "expressao": expr,
                "resposta": formatar_iso(alvo),
            }


def gerar_dataset(
    n_treino: int = 2600,
    n_val: int = 260,
    n_teste_variacao: int = 260,
    n_teste_inedita: int = 100,
    seed: int = 42,
    output_dir: Path | str = DIR_DATA,
    salvar: bool = True,
) -> dict[str, list[dict]]:
    """Gera proceduralmente os datasets particionados com gabarito exato."""
    random.seed(seed)
    output_path = Path(output_dir)
    vistos: set = set()

    # Cálculo da cota por família de treino
    qtd_fam_treino = len(FAMILIAS_TREINO)
    cota_treino = n_treino // qtd_fam_treino
    cota_val = n_val // qtd_fam_treino
    cota_teste_var = n_teste_variacao // qtd_fam_treino

    qtd_fam_inedita = len(FAMILIAS_INEDITAS)
    cota_teste_ined = n_teste_inedita // qtd_fam_inedita

    treino = []
    validacao = []
    teste = []

    # 1. Famílias de treino (variacao)
    for fam in FAMILIAS_TREINO:
        treino.extend([gerar_exemplo(fam, vistos) for _ in range(cota_treino)])
        validacao.extend([gerar_exemplo(fam, vistos) for _ in range(cota_val)])
        teste.extend([{**gerar_exemplo(fam, vistos), "tipo": "variacao"} for _ in range(cota_teste_var)])

    # 2. Famílias inéditas (somente no teste para avaliar generalização zero-shot)
    for fam in FAMILIAS_INEDITAS:
        teste.extend([{**gerar_exemplo(fam, vistos), "tipo": "inedita"} for _ in range(cota_teste_ined)])

    random.shuffle(treino)

    if salvar:
        output_path.mkdir(parents=True, exist_ok=True)
        for nome, lista in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
            caminho_arq = output_path / f"{nome}.jsonl"
            with open(caminho_arq, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in lista)

    return {"treino": treino, "validacao": validacao, "teste": teste}


def main():
    parser = argparse.ArgumentParser(description="Geração procedural de dataset sintético PT-BR -> ISO 8601")
    parser.add_argument("--treino", type=int, default=2600, help="Quantidade total de exemplos de treino (default: 2600)")
    parser.add_argument("--val", type=int, default=260, help="Quantidade total de exemplos de validação (default: 260)")
    parser.add_argument("--teste-variacao", type=int, default=260, help="Exemplos de teste em famílias conhecidas (default: 260)")
    parser.add_argument("--teste-inedita", type=int, default=100, help="Exemplos de teste em famílias inéditas (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade (default: 42)")
    parser.add_argument("--output-dir", type=str, default=str(DIR_DATA), help="Diretório de saída para os arquivos .jsonl")

    args = parser.parse_args()
    print("=" * 60)
    print("GERAÇÃO PROCEDURAL DE DATASET (PT-BR -> ISO 8601)")
    print("=" * 60)
    print(f"Diretório de destino: {args.output_dir}")
    print(f"Seed:                 {args.seed}")
    print(f"Configuração:         Treino={args.treino} | Val={args.val} | Teste={args.teste_variacao + args.teste_inedita}")

    dados = gerar_dataset(
        n_treino=args.treino,
        n_val=args.val,
        n_teste_variacao=args.teste_variacao,
        n_teste_inedita=args.teste_inedita,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    tipos_teste = Counter(e["tipo"] for e in dados["teste"])
    print("\nDataset gerado e salvo com sucesso:")
    print(f"  - Treino:    {len(dados['treino'])} exemplos")
    print(f"  - Validação: {len(dados['validacao'])} exemplos")
    print(f"  - Teste:     {len(dados['teste'])} exemplos ({tipos_teste['variacao']} variações + {tipos_teste['inedita']} inéditas)")
    print("=" * 60)


if __name__ == "__main__":
    main()
