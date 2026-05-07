#!/usr/bin/env python3
"""
principia_sfs_surgeon_v3.py
Extração de memória bruta usando decodificação URL-Safe tolerante a falhas do Principia/Google.
"""

import base64
import re
from pathlib import Path

# --- CAMINHO FIXO DA SUA SONDA ---
SFS_PATH = Path("/home/matheus/.steam/steam/steamapps/common/Kerbal Space Program/saves/JNSQ/principia_anchor.sfs")
OUT_BIN = Path("data/jnsq_gate0/principia_raw_state.bin")

def decode_payload(s: str) -> bytes:
    # 1. Remove qualquer lixo de espaçamento/quebra de linha
    compact = re.sub(r"\s+", "", s)
    
    # 2. Arruma o padding (enchimento) obrigatório do Base64
    missing_padding = len(compact) % 4
    if missing_padding != 0:
        compact += '=' * (4 - missing_padding)

    # 3. Usa o decodificador "URL Safe" que aceita os hífens (-) do Google/Principia
    try:
        return base64.urlsafe_b64decode(compact)
    except Exception as exc:
        raise ValueError(f"Falha total ao decodificar Base64: {exc}")

def main():
    if not SFS_PATH.exists():
        print(f"❌ Arquivo não encontrado: {SFS_PATH}")
        return

    text = SFS_PATH.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_principia = False
    brace_depth = 0
    chunks = []

    for line in lines:
        stripped = line.strip()
        
        # Identifica a entrada do nó do Principia
        if stripped == "name = PrincipiaPluginAdapter":
            in_principia = True
            brace_depth = 1  # Forçamos a profundidade pois já estamos dentro
            continue
            
        if in_principia:
            if stripped.startswith("serialized_plugin ="):
                _, value = stripped.split("=", 1)
                chunks.append(value.strip())
            
            # Controle de escopo
            brace_depth += stripped.count("{")
            brace_depth -= stripped.count("}")
            
            # Se o escopo do nó zerar, saímos
            if brace_depth <= 0 and "}" in stripped:
                break

    if not chunks:
        print("❌ Nenhuma linha 'serialized_plugin' extraída.")
        return

    full_payload = "".join(chunks)
    print(f"✅ Encontrados {len(chunks)} blocos. Tamanho total da string: {len(full_payload)} chars.")

    raw_bytes = decode_payload(full_payload)
    
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_BIN.write_bytes(raw_bytes)
    
    print(f"🚀 Sucesso! {len(raw_bytes)} bytes de física pura salvos em: {OUT_BIN}")

if __name__ == "__main__":
    main()