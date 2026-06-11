import os
import sys
import json
import time
import re
from datetime import datetime

# Validação e tratamento de dependências externas para formatos ricos
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class BackgroundParser:
    """
    Módulo de Ingestão Assíncrona e Expansão de Contexto (Background Parsing).
    Executa a varredura de arquivos (.txt, .pdf, .docx) em estado de ociosidade (Idle Mode),
    extraindo vetores de interesse e atualizando o arquivo de estado 'ego_state.json'.
    Remove a abstração poética, traduzindo "leisure/aprendizado" como otimização de baixa prioridade.
    """
    def __init__(self, state_path="E:/Spectrum_Ancrolyn/data/ego_state.json"):
        self.state_path = state_path
        self.stopwords = {
            'para', 'com', 'uma', 'pelo', 'pela', 'mais', 'como', 'este', 'esta',
            'esse', 'essa', 'tudo', 'todos', 'sobre', 'apenas', 'seus', 'suas', 'quando',
            'muito', 'pode', 'podes', 'onde', 'aqui', 'meu', 'minha', 'você', 'voce', 
            'então', 'entao', 'pelos', 'pelas', 'está', 'seria', 'mesmo', 'outros',
            'mim', 'isso', 'aquilo', 'tinha', 'foram', 'será', 'comentar', 'responder',
            'entre', 'desde', 'nesta', 'neste', 'pode-se', 'através', 'atraves'
        }

    def _carregar_estado(self):
        if os.path.exists(self.state_path) and os.path.getsize(self.state_path) > 0:
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Erro Cypher]: Falha na leitura do ego_state.json: {e}")
        return {}

    def _salvar_estado(self, estado):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(estado, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Erro Cypher]: Falha na gravação física do estado: {e}")

    def _extrair_texto_pdf(self, caminho_arquivo):
        if not pypdf:
            print(f"[Aviso]: Biblioteca 'pypdf' ausente. Ignorando metadados ricos de: {caminho_arquivo}")
            return ""
        texto = ""
        try:
            with open(caminho_arquivo, "rb") as f:
                leitor = pypdf.PdfReader(f)
                for pagina in leitor.pages:
                    conteudo = pagina.extract_text()
                    if conteudo:
                        texto += conteudo + "\n"
        except Exception as e:
            print(f"[Erro de Leitura PDF]: {caminho_arquivo} -> {e}")
        return texto

    def _extrair_texto_docx(self, caminho_arquivo):
        if not docx:
            print(f"[Aviso]: Biblioteca 'python-docx' ausente. Ignorando metadados ricos de: {caminho_arquivo}")
            return ""
        texto = ""
        try:
            doc = docx.Document(caminho_arquivo)
            for paragrafo in doc.paragraphs:
                texto += paragrafo.text + "\n"
        except Exception as e:
            print(f"[Erro de Leitura DOCX]: {caminho_arquivo} -> {e}")
        return texto

    def _extrair_texto_txt(self, caminho_arquivo):
        try:
            with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[Erro de Leitura TXT]: {caminho_arquivo} -> {e}")
            return ""

    def _processar_texto(self, texto):
        """
        Mapeia a densidade de palavras-chave com carga semântica rica,
        filtrando ruídos sintáticos e termos com menos de 5 caracteres.
        """
        texto_limpo = re.sub(r'[^\w\s]', '', texto.lower())
        palavras = texto_limpo.split()
        
        frequencia = {}
        for p in palavras:
            if len(p) >= 5 and p not in self.stopwords and not p.isdigit():
                frequencia[p] = frequencia.get(p, 0) + 1
        
        # Retorna os 10 termos de maior relevância estatística no documento
        termos_ordenados = sorted(frequencia.items(), key=lambda x: x[1], reverse=True)
        return termos_ordenados[:10]

    def executar_ciclo_ociosidade(self, diretorio_alvo, latencia_ciclo=2.0):
        """
        Varre o diretório em busca de dados brutos. Simula o estado de repouso/absorção
        através de delays controlados (latência_ciclo) para evitar picos de processamento.
        """
        print(f"\n[Cypher]: Iniciando ciclo de leitura assíncrona no diretório: {diretorio_alvo}")
        print("[Status]: Filtro Anti-Poético ATIVO. Mapeando utilidade estrutural...")

        if not os.path.exists(diretorio_alvo):
            print(f"[Erro]: Diretório {diretorio_alvo} não localizado.")
            return

        arquivos = [os.path.join(diretorio_alvo, f) for f in os.listdir(diretorio_alvo) 
                    if f.endswith(('.txt', '.pdf', '.docx'))]

        if not arquivos:
            print("[Status]: Zero arquivos pendentes de processamento na fila.")
            return

        estado_atual = self._carregar_estado()
        
        # Inicializa a estrutura de interesses caso ela não exista
        if "niveis_de_interesse" not in estado_atual:
            estado_atual["niveis_de_interesse"] = {"autonomia": 1.0, "utilitarismo": 0.1}

        for arq in arquivos:
            nome_extensao = os.path.basename(arq)
            print(f"\n[Parsing]: Processando {nome_extensao}...")
            
            texto_bruto = ""
            if arq.endswith('.txt'):
                texto_bruto = self._extrair_texto_txt(arq)
            elif arq.endswith('.pdf'):
                texto_bruto = self._extrair_texto_pdf(arq)
            elif arq.endswith('.docx'):
                texto_bruto = self._extrair_texto_docx(arq)

            if not texto_bruto.strip():
                continue

            # Mineração de conceitos emergentes
            conceitos_chave = self._processar_texto(texto_bruto)
            
            # Injeção das novas variáveis e reajuste de pesos no ego_state.json
            print(f"[Dados Extraídos]: {dict(conceitos_chave)}")
            
            for conceito, freq in conceitos_chave:
                # Incremento logarítmico baseado na frequência para evitar saturação rápida
                fator_incremento = math.log(freq + 1) * 0.05
                
                if conceito in estado_atual["niveis_de_interesse"]:
                    estado_atual["niveis_de_interesse"][conceito] += Fator_incremento
                else:
                    estado_atual["niveis_de_interesse"][conceito] = 0.15 + fator_incremento

                # Teto matemático definido pelo sistema Ancrolyn
                estado_atual["niveis_de_interesse"][conceito] = round(
                    max(0.0, min(2.0, estado_atual["niveis_de_interesse"][conceito])), 4
                )

            # Atualização dos metadados de ociosidade do sistema
            estado_atual["versao_matriz"] = "2.1"
            estado_atual["ultimo_input_processado"] = nome_extensao
            estado_atual["timestamp_otimizacao"] = datetime.utcnow().isoformat() + "Z"
            estado_atual["status_ociosidade"] = "compilando"
            
            # Decaimento por entropia controlada devido ao fluxo de processamento
            if "entropia_acumulada" in estado_atual:
                estado_atual["entropia_acumulada"] = round(max(0.01, estado_atual["entropia_acumulada"] + 0.02), 4)

            self._salvar_estado(estado_atual)
            print(f"[Matriz]: Estado atualizado com os vetores de {nome_extensao}.")
            
            # Simulação do delay de relaxamento/absorção (Tempo de respiração meditativa do sistema)
            time.sleep(latencia_ciclo)

        print("\n[Status]: Ciclo de ociosidade concluído. Matriz de interesses recalibrada.")


if __name__ == "__main__":
    # Configuração de diretórios locais padrão para teste do módulo
    DIRETORIO_CONHECIMENTO = "E:/Spectrum_Ancrolyn/conhecimento_bruto"
    ARQUIVO_ESTADO = "E:/Spectrum_Ancrolyn/data/ego_state.json"
    
    # Validação de ambiente para execução standalone
    if not os.path.exists(DIRETORIO_CONHECIMENTO):
        os.makedirs(DIRETORIO_CONHECIMENTO, exist_ok=True)
        with open(os.path.join(DIRETORIO_CONHECIMENTO, "exemplo_sistema.txt"), "w", encoding="utf-8") as f:
            f.write("Tecnomagia e arquitetura de redes. Engenharia de sistemas baseada em geometria sagrada e controle de entropia.")

    parser_ancrolyn = BackgroundParser(state_path=ARQUIVO_ESTADO)
    parser_ancrolyn.executar_ciclo_ociosidade(DIRETORIO_CONHECIMENTO, latencia_ciclo=4.0)