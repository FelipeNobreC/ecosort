"""Pré-processamento de texto em PT-BR para o reconhecimento de intenções.

Etapas: minúsculas -> remoção de acentos -> remoção de pontuação -> remoção de stopwords.

Observação importante: a lista de stopwords é PRÓPRIA (e não a do NLTK) porque
palavras como "onde", "como", "antes", "não" e "pode" costumam estar nas listas
padrão, mas são justamente as pistas mais fortes de intenção neste domínio.
"""
from __future__ import annotations

import re
import unicodedata

# Stopwords sem acento (aplicadas DEPOIS da remoção de acentos).
STOPWORDS: frozenset[str] = frozenset("""
a o as os um uma uns umas de da do das dos d em no na nos nas num numa
e eu tu ele ela nos vos eles elas me te se lhe lhes meu minha meus minhas
seu sua seus suas esse essa esses essas este esta estes estas isto isso aquilo
aquele aquela aqui ai la ali entao tambem ja so mesmo muito mais
ao aos pelo pela pelos pelas por para pra pro com
que q eh ser sou estou esta tem ter
voce vc mano vei cara parada hein ne tipo
""".split())

_RE_NAO_ALFANUM = re.compile(r"[^a-z0-9\s]")
_RE_ESPACOS = re.compile(r"\s+")


def remover_acentos(texto: str) -> str:
    """Remove diacríticos (ex.: 'reciclável' -> 'reciclavel')."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def preprocessar(texto: str, remover_stopwords: bool = True) -> str:
    """Normaliza o texto para o vetorizador TF-IDF."""
    t = remover_acentos(str(texto).lower())
    t = _RE_NAO_ALFANUM.sub(" ", t)
    tokens = _RE_ESPACOS.sub(" ", t).strip().split()
    if remover_stopwords:
        tokens = [tk for tk in tokens if tk not in STOPWORDS]
    return " ".join(tokens)
