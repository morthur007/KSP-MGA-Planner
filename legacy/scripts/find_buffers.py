import subprocess
import math
import csv

def test_buffer(b):
    """Testa um valor de buffer (em dias) e retorna se o Principia sobreviveu."""
    
    # 1. Gera os arquivos de entrada com o buffer testado
    subprocess.run([
        "python", "lambert_candidate_to_particle_leg_v0_1.py",
        "--candidate-csv", "data/mga_smoke/kekj_lambert_w25_parallel_merged.csv",
        "--rank", "1",
        "--bsp", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.bsp",
        "--tpc", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.ids.tpc",
        "--metadata", "data/spice_jnsq_v0_3/jnsq_principia_native_v0_3_1000y_1h_deg9.metadata.json",
        "--body-catalog", "data/jnsq_gate0/ksp_future_paused/body_catalog.json",
        "--central-body", "Sun",
        "--sequence", "Kerbin", "Eve", "Kerbin", "Jool",
        "--leg", "1",
        "--start-buffer-days", str(b),
        "--end-buffer-days", str(b),
        "--output-input-csv", "data/mga_smoke/validator_inputs/rank1_leg1_input.csv",
        "--output-expected-csv", "data/mga_smoke/validator_inputs/rank1_leg1_expected.csv",
        "--spice-to-principia-transform", "+Z,-X,+Y"  # A Matriz Campeã
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Chama o motor do Principia
    subprocess.run([
        "principia_particle_validator",
        "data/jnsq_gate0/principia_serialized_plugin.b64",
        "data/mga_smoke/validator_inputs/rank1_leg1_input.csv",
        "data/mga_smoke/validator_inputs/rank1_leg1_output.csv"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Analisa o resultado
    try:
        with open("data/mga_smoke/validator_inputs/rank1_leg1_output.csv") as f:
            o = next(csv.DictReader(f))
        
        if o["status"] != "ok":
            return False, float('inf')  # C++ falhou

        with open("data/mga_smoke/validator_inputs/rank1_leg1_expected.csv") as f:
            e = next(csv.DictReader(f))
            
        pos_err = math.sqrt(sum((float(o[k])-float(e[k]))**2 for k in ["x_m","y_m","z_m"])) / 1000.0
        
        # Se a nave foi estilingada numa velocidade absurda (erro > 1 milhão de km)
        # consideramos que a integração falhou fisicamente por estar muito perto.
        if pos_err > 1000000:
            return False, pos_err
            
        return True, pos_err
        
    except Exception:
        return False, float('inf')

def main():
    # Espaço de busca (em dias)
    low = 0.001  # Muito perto (com certeza vai explodir)
    high = 15.0  # Muito seguro
    tolerance = 0.05  # Precisão desejada (1.2 horas de erro)
    
    best_safe_buffer = high
    best_error = 0.0
    
    print("Iniciando Busca Binária do Buffer Limite...")
    print(f"Range: [{low} a {high}] dias | Tolerância: {tolerance} dias\n")
    
    step = 1
    while (high - low) > tolerance:
        mid = (low + high) / 2.0
        print(f"[{step:02d}] Testando limite = {mid:6.3f} dias...", end=" ")
        
        is_safe, err_km = test_buffer(mid)
        
        if is_safe:
            print(f"✅ SEGURO (Desvio: {err_km:10.2f} km). Tentando espremer mais...")
            best_safe_buffer = mid
            best_error = err_km
            high = mid  # Se é seguro, o limite verdadeiro está ABAIXO do mid
        else:
            print(f"❌ CRASH (Singularidade). Aumentando distância...")
            low = mid   # Se explodiu, precisamos de um buffer MAIOR que o mid
            
        step += 1

    print("\n" + "="*50)
    print("🚀 BUSCA CONCLUÍDA NA VELOCIDADE DA LUZ")
    print("="*50)
    print(f"Menor buffer de sobrevivência : {best_safe_buffer:.3f} dias")
    print(f"Desvio N-Corpos real capturado: {best_error:.2f} km")

if __name__ == "__main__":
    main()