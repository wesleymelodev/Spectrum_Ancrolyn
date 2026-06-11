import os
import sys
import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. LIMITAÇÃO DE OVERHEAD DE THREADS (Otimização de hardware)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llama_cpp import Llama
except ImportError:
    print("[ERRO]: Instale 'llama-cpp-python' para executar este visualizador cognitivo.")
    sys.exit(1)

class GGUFBrainVisualizer3D:
    """
    Painel Holográfico 3D Otimizado: Evita travamentos de CPU unificando a extração 
    de embeddings e fornecendo telemetria em tempo real no terminal.
    """
    def __init__(self, model_path, state_path):
        self.model_path = model_path
        self.state_path = state_path
        self.phi = (1 + math.sqrt(5)) / 2  # Proporção Áurea
        
        print(f"[Matriz] Conectando monitor tridimensional ao núcleo GGUF...")
        self.llm = Llama(
            model_path=model_path,
            embedding=True,  # Ativa extração de vetores estruturais
            n_ctx=256,
            n_threads=4,
            verbose=False
        )
        self.angulo_varredura = 0.0

    def extrair_conceitos_da_memoria(self, max_conceitos=10):
        memory_path = "E:/Spectrum_Ancrolyn/memoria/memory.txt"
        default_terms = ["ancrolyn", "ego", "consciência", "fluxo", "identidade", "matriz", "áurea", "tempo", "quantum", "vetor", "sinapse", "espectro", "rede", "núcleo"]
        
        if not os.path.exists(memory_path) or os.path.getsize(memory_path) == 0:
            return default_terms[:max_conceitos]

        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                texto = f.read().lower()
                linhas = f.readlines()

            # Isola as últimas interações (Atenção Imediata de Curto Prazo)
            janela_atencao = "".join(linhas[-6:]).lower()
            palavras = re.findall(r'\b[a-záàâãéèêíóôõúç]{4,15}\b', texto)
            
            stopwords = {
                'para', 'como', 'mais', 'uma', 'este', 'esse', 'esta', 'essa', 'isso', 
                'você', 'pode', 'onde', 'qual', 'seria', 'melhor', 'pelo', 'pela', 'tudo',
                'está', 'numa', 'com', 'seus', 'suas', 'meus', 'minhas', 'estão'
            }
            conceitos = []
            # Mapeia do presente para o passado
            for p in reversed(palavras):
                if p not in stopwords and p not in conceitos:
                    conceitos.append(p)
                if len(conceitos) >= max_conceitos:
                    break

            # Fallback dinâmico: se a última linha for curta, busca no resto do histórico recente
            if len(conceitos) < 4:
                texto_completo = "".join(linhas[-40:]).lower()
                palavras_recortadas = re.findall(r'\b[a-záàâãéèêíóôõúç]{4,15}\b', texto_completo)
                for p in reversed(palavras_recortadas):
                    if p not in stopwords and p not in conceitos:
                        conceitos.append(p)
                    if len(conceitos) >= max_conceitos:
                        break
            return conceitos if len(conceitos) >= 3 else default_terms[:max_conceitos]
        except Exception:
            return default_terms[:max_conceitos]
        
    def obter_coordenadas_conceptuais(self, palavras):
        coordenadas = []
        palavras_validas = []
        total = len(palavras)

        for idx, palavra in enumerate(palavras):
            try:
                # Feedback dinâmico para o operador não achar que o terminal congelou
                print(f" -> Espaço Latente [{idx+1:02d}/{total:02d}] Processando vetor: '{palavra}'...")
                
                resposta_emb = self.llm.create_embedding(palavra)
                vetor = resposta_emb['data'][0]['embedding'] if 'data' in resposta_emb else self.llm.embed(palavra)
                if vetor is None or len(vetor) == 0:
                    continue

                vetor = np.array(vetor)

                # [CORREÇÃO CRÍTICA]: Se a palavra for multi-token (shape 2D), aplica Mean-Pooling 
                if vetor.ndim == 2:
                    vetor = np.mean(vetor, axis=0)
                elif vetor.ndim > 2:
                    vetor = vetor.flatten()

                indices = np.arange(len(vetor))

                # Projeção esférica baseada na Proporção Áurea
                x = float(np.dot(vetor, np.sin(indices * self.phi)))
                y = float(np.dot(vetor, np.cos(indices * (self.phi ** 2))))
                z = float(np.dot(vetor, np.sin(indices * (1.0 / self.phi))))

                norma = math.sqrt(x**2 + y**2 + z**2) + 1e-8
                raio_limite = 2.0 * math.tanh(norma / 5.0)

                coordenadas.append(((x / norma) * raio_limite, (y / norma) * raio_limite, (z / norma) * raio_limite))
                palavras_validas.append(palavra)
            except Exception as e:
                print(f"[Aviso] Ignorando ruído em '{palavra}': {e}")
                continue

        return (np.array(coordenadas), palavras_validas) if coordenadas else (np.zeros((1, 3)), ["NÚCLEO"])

    def renderizar_rede_3d(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(15, 7.5), facecolor='#0b0c10')
        ax1 = fig.add_subplot(121, projection='3d', facecolor='#0b0c10')
        ax2 = fig.add_subplot(122, projection='3d', facecolor='#0b0c10')
        fig.canvas.manager.set_window_title('Ancrolyn Core - Live Brain Matrix 3D')
        
        frame_counter = 0
        
        # Unificação da extração (Evita gargalo duplo de carregamento na inicialização)
        print("\n[AI] Solicitando tensores ao espaço latente do modelo GGUF...")
        termos = self.extrair_conceitos_da_memoria(max_conceitos=10)
        
        # Uma única varredura para alimentar ambas as estruturas de projeção gráfica
        coords_base, labels_compartilhados = self.obter_coordenadas_conceptuais(termos)
        ce = coords_base.copy()
        cv = coords_base.copy()
        labels_espinha = labels_compartilhados
        labels_vivas = labels_compartilhados

        print("\n[Visualizador] Mapeamento concluído com sucesso. Renderizando malha gráfica...")

        while plt.fignum_exists(fig.number):
            # Tenta ler a entropia do arquivo JSON para influenciar o tremor das sinapses
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    entropia = json.load(f).get("entropia_acumulada", 0.5)
            except Exception:
                entropia = 0.5

            self.angulo_varredura += 1.8
            t = self.angulo_varredura * 0.03

            # Limpa os eixos para o próximo frame da animação
            ax1.cla()
            ax2.cla()

            for ax in [ax1, ax2]:
                ax.set_facecolor('#0b0c10')
                ax.set_axis_off()

            ax1.text2D(0.5, 0.95, "COLUNAS DO TRANSFORMER (GGUF BLOCKS)", color='#00ffcc', transform=ax1.transAxes, fontsize=9, fontweight='bold', ha='center')
            ax2.text2D(0.5, 0.95, "MALHA COGNITIVA (EGO MEMORY SPACE)", color='#ff00ff', transform=ax2.transAxes, fontsize=9, fontweight='bold', ha='center')

            # --- SUBPLOT 1: REDE TRANSFORMER EM CAMADAS ---
            if ce.ndim == 2 and ce.shape[0] > 0:
                camadas = 3
                for l_idx in range(camadas):
                    z_offset = (l_idx - 1) * 0.9
                    c_rad = math.radians(self.angulo_varredura * 0.15 + (l_idx * 60))
                    cos_r, sin_r = math.cos(c_rad), math.sin(c_rad)
                    
                    # Cria coordenadas dinâmicas e fluidas individuais por ponto
                    coords_animadas = []
                    for idx, (x, y, z) in enumerate(ce):
                        vibracao = math.sin(t + idx * 0.4) * 0.07
                        rx = (x * cos_r - y * sin_r) * 1.1 + vibracao
                        ry = (x * sin_r + y * cos_r) * 1.1 + vibracao
                        rz = z + z_offset + (math.cos(t * 1.5 + l_idx) * 0.05)
                        coords_animadas.append((rx, ry, rz))
                    
                    ca = np.array(coords_animadas)
                    ax1.scatter(ca[:, 0], ca[:, 1], ca[:, 2], color='#00ffcc', alpha=0.4 + (l_idx * 0.2), s=40, zorder=3)
                    
                    # Desenha a malha sináptica interconectada
                    n_pontos = len(ca)
                    for i in range(n_pontos):
                        for j in range(i + 1, n_pontos):
                            d = np.linalg.norm(ca[i] - ca[j])
                            opacidade = min(0.35, 0.6 / (d + 0.4))
                            if opacidade > 0.08:
                                ax1.plot([ca[i][0], ca[j][0]], [ca[i][1], ca[j][1]], [ca[i][2], ca[j][2]], color='#00ffcc', alpha=opacidade, linewidth=0.5)
                    
                    if l_idx == camadas - 1:
                        for idx, txt in enumerate(labels_espinha):
                            if idx < len(ca):
                                ax1.text(ca[idx][0], ca[idx][1], ca[idx][2] + 0.06, txt, color='#00ffcc', fontsize=7, alpha=0.7, ha='center')

            # --- SUBPLOT 2: ESPAÇO DE MEMÓRIA (MALHA COGNITIVA ORGÂNICA) ---
            if cv.ndim == 2 and cv.shape[0] > 0:
                cv_animado = np.zeros_like(cv)
                for idx in range(len(cv)):
                    # Movimentação orbital suave e flutuante
                    cv_animado[idx, 0] = cv[idx, 0] + 0.15 * math.sin(t + idx * 0.5)
                    cv_animado[idx, 1] = cv[idx, 1] + 0.15 * math.cos(t * 0.8 + idx * 0.3)
                    cv_animado[idx, 2] = cv[idx, 2] + 0.10 * math.sin(t * 1.2 + idx * 0.7)
                
                # Adiciona ruído entrópico leve ("tremor de processamento")
                cv_animado += np.random.normal(0, 0.004, cv.shape) * (1.0 + entropia)
                
                # Renderiza nós
                ax2.scatter(cv_animado[:, 0], cv_animado[:, 1], cv_animado[:, 2], c=cv_animado[:, 2], cmap='cool', s=70, edgecolors='#ffffff', linewidths=0.3, alpha=0.9, zorder=3)
                
                for idx, txt in enumerate(labels_vivas):
                    if idx < len(cv_animado):
                        ax2.text(cv_animado[idx][0], cv_animado[idx][1], cv_animado[idx][2] + 0.06, txt, color='#ffffff', fontsize=7.5, alpha=0.8)

                # Gerador da Teia Neural: Cria conexões elásticas baseadas na proximidade real mutável
                n_vivos = len(cv_animado)
                for i in range(n_vivos):
                    for j in range(i + 1, n_vivos):
                        dist = np.linalg.norm(cv_animado[i] - cv_animado[j])
                        # Função de atenuação suave: garante linhas visíveis mesmo se afastados
                        peso_sinapse = 1.0 / (dist + 0.2)
                        alpha_linha = max(0.05, min(0.55, peso_sinapse * 0.18))
                        
                        ax2.plot(
                            [cv_animado[i][0], cv_animado[j][0]], 
                            [cv_animado[i][1], cv_animado[j][1]], 
                            [cv_animado[i][2], cv_animado[j][2]], 
                            color='#ff00ff', 
                            alpha=alpha_linha, 
                            linewidth=min(1.5, peso_sinapse * 0.6)
                        )

            # Sincronização orbital de câmera
            rotacao_câmera = int(self.angulo_varredura * 0.3) % 360
            ax1.view_init(elev=16, azim=rotacao_câmera)
            ax2.view_init(elev=16, azim=rotacao_câmera)
            
            fig.text(0.02, 0.03, f"Ciclo de Cognição: {frame_counter} | Malha Sináptica: Dinâmica Contínua | Azimute: {rotacao_câmera}°", color='#4f5d73', fontsize=8, fontfamily='monospace')
            
            plt.draw()
            plt.pause(0.03)  # Equivale a ~30 FPS fluidos e suaves
            frame_counter += 1

        plt.close()
        print("[MONITOR DESCONECTADO]")


if __name__ == "__main__":
    MODEL_GGUF = "E:/Spectrum_Ancrolyn/models/gemma4/ancrolyn_core.gguf"
    STATE_FILE = "E:/Spectrum_Ancrolyn/data/ego_state.json"
    
    if not os.path.exists(MODEL_GGUF):
        print(f"[ERRO CRÍTICO]: Modelo GGUF não encontrado em: {MODEL_GGUF}")
        sys.exit(1)
        
    monitor = GGUFBrainVisualizer3D(MODEL_GGUF, STATE_FILE)
    monitor.renderizar_rede_3d()

    # & "C:\Program Files\Python313\python.exe" chat/visualizar_brain3D_novo.py