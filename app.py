"""Interface Streamlit do EcoSort.

Execução:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.decisao.motor_regras import carregar_base  # noqa: E402
from src.pipeline import processar  # noqa: E402

NOMES_PT = {
    "cardboard": "Papelão", "glass": "Vidro", "metal": "Metal",
    "paper": "Papel", "plastic": "Plástico", "trash": "Rejeito",
}
INTENCOES_PT = {
    "como_descartar": "Como descartar", "e_reciclavel": "É reciclável?",
    "onde_levar": "Onde levar", "preparo_antes": "Preparo antes do descarte",
    "fora_do_escopo": "Fora do escopo",
}

st.set_page_config(page_title="EcoSort", page_icon="♻️", layout="wide")
st.title("♻️ EcoSort — Classificação de Resíduos com Visão Computacional e PLN")
st.caption("Envie uma foto do resíduo, faça uma pergunta em português e veja como descartá-lo corretamente.")

col_entrada, col_saida = st.columns([1, 1.3])

with col_entrada:
    arquivo = st.file_uploader("1. Foto do resíduo", type=["jpg", "jpeg", "png", "webp"])
    pergunta = st.text_input("2. Sua pergunta", value="como descarto isso?",
                             placeholder="ex.: isso é reciclável? / onde posso levar? / precisa lavar antes?")
    executar = st.button("Analisar", type="primary", disabled=arquivo is None)
    if arquivo is not None:
        st.image(Image.open(arquivo), caption="Imagem enviada", width="stretch")

with col_saida:
    if executar and arquivo is not None:
        with st.spinner("Analisando..."):
            r = processar(Image.open(arquivo), pergunta)
        v, p, d = r["visao"], r["pln"], r["decisao"]
        limiar = d["detalhes"]["limiar_visao"]

        st.subheader("Saídas intermediárias")
        c1, c2 = st.columns(2)
        c1.metric("Material (Visão)", NOMES_PT[v["classe"]], f"confiança {v['confianca']:.1%}",
                  delta_color="normal" if v["confianca"] >= limiar else "inverse")
        c2.metric("Intenção (PLN)", INTENCOES_PT[p["intencao"]], f"confiança {p['confianca']:.1%}")
        st.write("**Top-2 da visão:** " + " | ".join(f"{NOMES_PT[c]}: {pr:.1%}" for c, pr in v["top2"]))
        st.progress(min(1.0, v["confianca"]), text=f"Confiança da visão (limiar = {limiar:.0%})")

        st.subheader("Decisão (sistema baseado em regras)")
        st.code(f"{d['regra_id']}: {d['regra_descricao']}", language="text")
        if d["classe_considerada"]:
            mat = carregar_base()["materiais"][d["classe_considerada"]]
            st.write(f"Lixeira: **{mat['lixeira_cor'].upper()}** · Reciclável: "
                     f"**{'sim' if mat['reciclavel'] else 'não'}**")

        st.subheader("Resposta")
        (st.warning if d["precisa_nova_foto"] else st.success)(d["resposta"])

        with st.expander("JSON completo do fluxo"):
            st.json(r)
    else:
        st.info("Envie uma imagem e clique em **Analisar**.")

with st.sidebar:
    st.header("Como funciona")
    st.markdown(
        "1. **Visão** (MobileNetV2) classifica o material em 6 classes.\n"
        "2. **PLN** (TF-IDF + Regressão Logística) reconhece a intenção da pergunta.\n"
        "3. **Decisão** (regras SE-ENTÃO + base de conhecimento JSON) combina classe, "
        "confiança e intenção e gera a resposta — ou pede outra foto se a confiança for baixa."
    )
    st.header("Cores da coleta seletiva")
    for k, cor in carregar_base()["_meta"]["cores_coleta_seletiva"].items():
        st.write(f"- {k}: **{cor}**")
