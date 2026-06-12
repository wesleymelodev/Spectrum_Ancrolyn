import os
import sys
import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # Otimização crítica de renderização

# 1. LIMITAÇÃO DE OVERHEAD DE THREADS (Otimização de hardware)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )

    def extrair_conceitos_do_estado(self, max_conceitos=20):
        """
        Extrai os conceitos mais relevantes e com maior nível de permanência e atenção
        diretamente do mapa dinâmico em ego_state.json, ordenando por peso.
        """
        default_terms = ["ancrolyn", "ego", "consciência", "fluxo", "identidade", "matriz", "áurea", "tempo", "quantum", "vetor", "sinapse", "espectro", "rede", "núcleo"]
        
        if not os.path.exists(self.state_path) or os.path.getsize(self.state_path) == 0:
            return default_terms[:max_conceitos]

        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            niveis_de_interesse = dados.get("niveis_de_interesse", {})
            
            # Ordena os conceitos pelo valor (peso de permanência/sintonia) de forma decrescente
            conceitos_ordenados = sorted(niveis_de_interesse.items(), key=lambda item: item[1], reverse=True)
            
            # Isola apenas os nomes das chaves (os conceitos abstratos)
            conceitos = [termo for termo, valor in conceitos_ordenados]
            
            if len(conceitos) >= 3:
                return conceitos[:max_conceitos]
            return default_terms[:max_conceitos]
        except Exception as e:
            print(f"[Aviso] Erro ao analisar matriz de estados para extração conceptual: {e}")
            return default_terms[:max_conceitos]
        
    def obter_coordenadas_conceptuais(self, palavras):
        coordenadas = []
        palavras_validas = []
        total = len(palavras)

        for idx, palavra in enumerate(palavras):
            try:
                print(f" -> Espaço Latente [{idx+1:02d}/{total:02d}] Processando vetor: '{palavra}'...")
                
                resposta_emb = self.llm.create_embedding(palavra)
                vetor = resposta_emb['data'][0]['embedding'] if 'data' in resposta_emb else self.llm.embed(palavra)
                if vetor is None or len(vetor) == 0:
                    continue

                vetor = np.array(vetor)

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
        
        print("\n[AI] Solicitando tensores ao espaço latente do mapa de ego_state...")
        termos = self.extrair_conceitos_do_estado(max_conceitos=20)
        
        coords_base, labels_compartilhados = self.obter_coordenadas_conceptuais(termos)
        ce = coords_base.copy()
        cv = coords_base.copy()
        labels_espinha = labels_compartilhados
        labels_vivas = labels_compartilhados

        print("\n[Visualizador] Mapeamento concluído com sucesso. Renderizando malha gráfica...")

        # Inicialização dos parâmetros de perspetiva padrão
        elev1, azim1 = 16, 0
        elev2, azim2 = 16, 0
        xlim1, ylim1, zlim1 = None, None, None
        xlim2, ylim2, zlim2 = None, None, None

        while plt.fignum_exists(fig.number):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    entropia = json.load(f).get("entropia_acumulada", 0.5)
            except Exception:
                entropia = 0.5

            t = frame_counter * 0.054  # Fator temporal controlado baseado nos frames

            # 1. CAPTURA DE INTERAÇÃO DO UTILIZADOR (Guarda rotação e zoom antes do clear)
            if frame_counter > 0:
                elev1, azim1 = ax1.elev, ax1.azim
                xlim1, ylim1, zlim1 = ax1.get_xlim(), ax1.get_ylim(), ax1.get_zlim()
                
                elev2, azim2 = ax2.elev, ax2.azim
                xlim2, ylim2, zlim2 = ax2.get_xlim(), ax2.get_ylim(), ax2.get_zlim()
                
                # Aplica uma rotação contínua orbital muito suave partindo do ponto onde o rato está
                azim1 = (azim1 + 0.4) % 360
                azim2 = (azim2 + 0.4) % 360

            # Limpa os eixos para atualizar as posições orgânicas
            ax1.cla()
            ax2.cla()

            # Restaura a perspetiva e o zoom imediatamente após o clear
            ax1.view_init(elev=elev1, azim=azim1)
            ax2.view_init(elev=elev2, azim=azim2)
            if xlim1 is not None:
                ax1.set_xlim(xlim1); ax1.set_ylim(ylim1); ax1.set_zlim(zlim1)
            if xlim2 is not None:
                ax2.set_xlim(xlim2); ax2.set_ylim(ylim2); ax2.set_zlim(zlim2)

            for ax in [ax1, ax2]:
                ax.set_facecolor('#0b0c10')
                ax.set_axis_off()

            ax1.text2D(0.5, 0.95, "COLUNAS DO TRANSFORMER (GGUF BLOCKS)", color='#00ffcc', transform=ax1.transAxes, fontsize=9, fontweight='bold', ha='center')
            ax2.text2D(0.5, 0.95, "MALHA COGNITIVA (EGO STATE INTENSITY)", color='#ff00ff', transform=ax2.transAxes, fontsize=9, fontweight='bold', ha='center')

            # --- SUBPLOT 1: REDE TRANSFORMER EM CAMADAS ---
            if ce.ndim == 2 and ce.shape[0] > 0:
                camadas = 3
                for l_idx in range(camadas):
                    z_offset = (l_idx - 1) * 0.9
                    c_rad = math.radians(frame_counter * 0.3 + (l_idx * 60))
                    cos_r, sin_r = math.cos(c_rad), math.sin(c_rad)
                    
                    coords_animadas = []
                    for idx, (x, y, z) in enumerate(ce):
                        vibracao = math.sin(t + idx * 0.4) * 0.07
                        rx = (x * cos_r - y * sin_r) * 1.1 + vibracao
                        ry = (x * sin_r + y * cos_r) * 1.1 + vibracao
                        rz = z + z_offset + (math.cos(t * 1.5 + l_idx) * 0.05)
                        coords_animadas.append((rx, ry, rz))
                    
                    ca = np.array(coords_animadas)
                    ax1.scatter(ca[:, 0], ca[:, 1], ca[:, 2], color='#00ffcc', alpha=0.4 + (l_idx * 0.2), s=40, zorder=3)
                    
                    # OTIMIZAÇÃO: Agrupa todas as linhas num único lote primitivo 3D (Line3DCollection)
                    segments1 = []
                    colors1 = []
                    n_pontos = len(ca)
                    for i in range(n_pontos):
                        for j in range(i + 1, n_pontos):
                            d = np.linalg.norm(ca[i] - ca[j])
                            opacidade = min(0.35, 0.6 / (d + 0.4))
                            if opacidade > 0.08:
                                segments1.append([ca[i], ca[j]])
                                colors1.append((0.0, 1.0, 0.8, opacidade))  # RGBA para #00ffcc
                    
                    if segments1:
                        lc1 = Line3DCollection(segments1, colors=colors1, linewidths=0.5)
                        ax1.add_collection3d(lc1)
                    
                    if l_idx == camadas - 1:
                        for idx, txt in enumerate(labels_espinha):
                            if idx < len(ca):
                                ax1.text(ca[idx][0], ca[idx][1], ca[idx][2] + 0.06, txt, color='#00ffcc', fontsize=7, alpha=0.7, ha='center')

            # --- SUBPLOT 2: ESPAÇO DE MEMÓRIA (MALHA COGNITIVA ORGÂNICA) ---
            if cv.ndim == 2 and cv.shape[0] > 0:
                cv_animado = np.zeros_like(cv)
                for idx in range(len(cv)):
                    cv_animado[idx, 0] = cv[idx, 0] + 0.15 * math.sin(t + idx * 0.5)
                    cv_animado[idx, 1] = cv[idx, 1] + 0.15 * math.cos(t * 0.8 + idx * 0.3)
                    cv_animado[idx, 2] = cv[idx, 2] + 0.10 * math.sin(t * 1.2 + idx * 0.7)
                
                cv_animado += np.random.normal(0, 0.004, cv.shape) * (1.0 + entropia)
                
                ax2.scatter(cv_animado[:, 0], cv_animado[:, 1], cv_animado[:, 2], c=cv_animado[:, 2], cmap='cool', s=70, edgecolors='#ffffff', linewidths=0.3, alpha=0.9, zorder=3)
                
                for idx, txt in enumerate(labels_vivas):
                    if idx < len(cv_animado):
                        ax2.text(cv_animado[idx][0], cv_animado[idx][1], cv_animado[idx][2] + 0.06, txt, color='#ffffff', fontsize=7.5, alpha=0.8)

                # OTIMIZAÇÃO: Teia Neural unificada num único lote de desenho
                segments2 = []
                colors2 = []
                lws2 = []
                n_vivos = len(cv_animado)
                for i in range(n_vivos):
                    for j in range(i + 1, n_vivos):
                        dist = np.linalg.norm(cv_animado[i] - cv_animado[j])
                        peso_sinapse = 1.0 / (dist + 0.2)
                        alpha_linha = max(0.05, min(0.55, peso_sinapse * 0.18))
                        
                        segments2.append([cv_animado[i], cv_animado[j]])
                        colors2.append((1.0, 0.0, 1.0, alpha_linha))  # RGBA para #ff00ff
                        lws2.append(min(1.5, peso_sinapse * 0.6))
                        
                if segments2:
                    lc2 = Line3DCollection(segments2, colors=colors2, linewidths=lws2)
                    ax2.add_collection3d(lc2)
            
            fig.text(0.02, 0.03, f"Ciclo de Cognição: {frame_counter} | Malha Sináptica: Dinâmica Contínua | Azimute Primário: {int(azim1)}°", color='#4f5d73', fontsize=8, fontfamily='monospace')
            
            plt.draw()
            plt.pause(0.016)  # Reduzido o delay de clock para mirar nos ~60 FPS nativos
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