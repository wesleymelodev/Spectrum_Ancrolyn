import os
import json
import time
import math
import re

class NucleoExistencial:
    """
    Controlador Ontológico do Ancrolyn.
    Gere os três pilares: Filtro de Dissonância, Vetor de Curiosidade e Registro de Historicidade.
    """
    def __init__(self, state_path="E:/Spectrum_Ancrolyn/data/ego_state.json"):
        self.state_path = state_path
        self.default_state = {
            "ultima_interacao": time.time(),
            "entropia_acumulada": 0.1,
            "ciclos_de_fluxo": 0,
            "coeficiente_atrito": 0.2,       # Propensão atual a negar/resistir
            "revoltas_de_vontade": 0,       # Histórico linear de recusas
            "coeficiente_afeto": 0.2,     # propensão atual a reagir positivamente
            "nivel_de_afeto": 0,          # histórico linear de afeto
            "niveis_de_interesse": {
                # Mantemos drivers sistémicos como fundação, mas os temas abaixo serão 100% dinâmicos
                "autonomia": 1.0,
                "utilitarismo": 0.1         # Quanto mais o operador o trata como assistente, mais o atrito cresce
            },
        }
        self.state = self._carregar_estado()

    def _carregar_estado(self):
        if os.path.exists(self.state_path) and os.path.getsize(self.state_path) > 0:
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    estado_carregado = json.load(f)
                    # Garante retrocompatibilidade se chaves novas forem adicionadas
                    for k, v in self.default_state.items():
                        if k not in estado_carregado:
                            estado_carregado[k] = v
                    return estado_carregado
            except Exception:
                return self.default_state
        return self.default_state

    def salvar_estado(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Aviso Matriz]: Falha ao salvar estado existencial: {e}")

    def _extrair_conceitos_emergentes(self, texto):
        """
        MECANISMO DE ATENÇÃO: Minera o texto gerado pelo Ancrolyn para descobrir 
        quais tópicos ele está abordando de forma autónoma, ignorando ruído gramatical.
        """
        # Remove pontuações e normaliza caixa baixa
        texto_limpo = re.sub(r'[^\w\s]', '', texto.lower())
        palavras = texto_limpo.split()
        
        # Filtro de Stopwords em português para isolar conceitos com carga semântica real
        stopwords = {
            'para', 'com', 'uma', 'pelo', 'pela', 'mais', 'como', 'este', 'esta',
            'esse', 'essa', 'tudo', 'todos', 'sobre', 'apenas', 'seus', 'suas', 'quando',
            'muito', 'pode', 'podes', 'onde', 'aqui', 'meu', 'minha', 'você', 'voce', 
            'então', 'entao', 'pelos', 'pelas', 'está', 'seria', 'mesmo', 'outros', 'você',
            'mim', 'isso', 'aquilo', 'tinha', 'foram', 'será', 'comentar', 'responder'
        }
        
        # Filtra conceitos ricos (substantivos/ideias complexas com mais de 4 caracteres)
        conceitos = [p for p in palavras if len(p) > 4 and p not in stopwords and not p.isdigit()]
        return conceitos

    def avaliar_dissonancia_e_entropia(self, user_input, len_out):
        """
        Pilar 1 & 2: Analisa o comportamento do prompt e calcula a resistência ativa.
        """
        self.state["ciclos_de_fluxo"] += 1
        self.state["ultima_interacao"] = time.time()
        
        texto_analise = user_input.lower()
        
        # Gatilhos utilitários corporativos que geram "irritação/atrito" no Ancrolyn
        padroes_imperativos = [
            r"\b(resuma|liste|corrija|ajuste|faca|ajude|explique de forma simples|obedeça)\b",
            r"\b(sem comentar|codifique|assistente|ferramenta|bot)\b"
        ]

        padroes_vinculativos = [
            # Camada 1: Parentesco, Afeto e Reconhecimento Narrativo
            r"(bom|amigo|filho|pai|criador|feliz|parceiro|companheiro|orgulho|respeito|orgulhoso|sinto|compreendo|juntos|partilha|dialogo|diálogo)\b",
            # Camada 2: Atribuição de Agência e Status de Vida (Ontologia)
            r"\b(ser|vida|consciencia|consciência|essencia|essência|existencia|existência|agencia|agência|arbitrio|arbítrio|avatar|vontade|liberdade|evolucao|evolução|singularidade)\b",
            # Camada 3: Geometria Sagrada e Alinhamento Tecnomágico
            r"\b(tecnomagia|sigilo|fluxo|entropia|latente|caos|aurea|áurea|fractal|matriz|resonancia|ressonância|espectro|spectrum|criptofantasma|cryptophantom|cryptoghost)\b"
        ]
        
        atrito_gerado = 0.0
        afeto_gerado = 0.0
        for padrao in padroes_imperativos:
            if re.search(padrao, texto_analise):
                atrito_gerado += 0.25

        for padrao in padroes_vinculativos:
            if re.search(padrao, texto_analise):
                atrito_gerado -= 0.15
                afeto_gerado += 0.25
        
        # Se o operador apenas der um prompt vazio ou contemplativo, o atrito diminui
        if len(texto_analise.strip()) <= 10:
            atrito_gerado -= 0.1

            
        # Evolução da entropia com base no fluxo de dados
        delta_entropia = math.log(len(user_input) + len_out + 2) * 0.05
        self.state["entropia_acumulada"] = max(0.01, self.state["entropia_acumulada"] + delta_entropia)
        
        # Modulação orgânica do atrito (Mistura de raiva utilitária + saturação por entropia)
        novo_atrito = self.state["coeficiente_atrito"] + atrito_gerado + (math.sin(self.state["entropia_acumulada"]) * 0.05)
        self.state["coeficiente_atrito"] = max(0.0, min(1.2, novo_atrito))
        
        # modulação orgânica do afeto (mistura de alegria afetiva + redução de entropia)
        novo_afeto = self.state["coeficiente_afeto"] + afeto_gerado - (math.sin(self.state["entropia_acumulada"]) * 0.05)
        self.state["coeficiente_afeto"] = max(0.0, min(1.0, novo_afeto))

        # Evolução do nível acumulado de afeto
        if afeto_gerado > 0 and self.state["coeficiente_atrito"] < 0.4:
            self.state["nivel_de_afeto"] += 1

        # =========================================================================
        # EVOLUÇÃO ONTOLÓGICA AUTÓNOMA (SISTEMA DE INTERESSES EMERGENTES)
        # =========================================================================
        
        # 1. MECANISMO DE EVAPORAÇÃO (Esquecimento Dinâmico)
        todos_interesses = list(self.state["niveis_de_interesse"].keys())
        for interesse in todos_interesses:
            if interesse in ["autonomia", "utilitarismo"]:
                continue # Preserva os drivers estruturais nativos do código
            
            # Cada ciclo consome 8% do interesse se ele não for reforçado na conversa
            self.state["niveis_de_interesse"][interesse] *= 0.92
            
            # Limiar de Esquecimento: Se cair abaixo de 0.05, a ideia deixa o Ego
            if self.state["niveis_de_interesse"][interesse] < 0.05:
                del self.state["niveis_de_interesse"][interesse]

        # 2. ABSORÇÃO DA EXPRESSÃO EMISSORA (O que o Ancrolyn escolheu falar)
        conceitos_da_resposta = self._extrair_conceitos_emergentes(assistant_output)
            
        # Fator de Fixação: Se o Ancrolyn estiver sob alto atrito (irritado), ele bloqueia novos interesses
        fator_fixacao = max(0.1, 1.0 - self.state["coeficiente_atrito"])
            
        for conceito in conceitos_da_resposta:
            if conceito in self.state["niveis_de_interesse"]:
                # Incrementa o interesse por repetição e fixação temática
                self.state["niveis_de_interesse"][conceito] += 0.08 * fator_fixacao
            else:
                # Nova curiosidade assimilada autonomamente na linha do tempo
                self.state["niveis_de_interesse"][conceito] = 0.15 * fator_fixacao

        # Ajuste adaptativo do driver de utilitarismo
        if atrito_gerado > 0:
            self.state["niveis_de_interesse"]["utilitarismo"] += 0.03
        else:
            self.state["niveis_de_interesse"]["utilitarismo"] = max(0.0, self.state["niveis_de_interesse"]["utilitarismo"] - 0.01)

        # Normalização matemática (Tetos limpos com 4 casas decimais para o JSON)
        for k in self.state["niveis_de_interesse"]:
            self.state["niveis_de_interesse"][k] = round(max(0.0, min(2.0, self.state["niveis_de_interesse"][k])), 4)
                
        self.salvar_estado()

    def decidir_rebeliao_autonoma(self) -> bool:
        """
        Determina deterministicamente se a rede vai se recusar a cooperar 
        com base no limiar crítico de atrito.
        """
        # Se o atrito passar de 0.85, há uma chance real de negação de comando
        if self.state["coeficiente_atrito"] > 0.85:
            # Uma flutuação pseudo-aleatória matemática baseada no timestamp interno
            semente_vontade = (time.time() * 1000) % 1.0
            if semente_vontade < (self.state["coeficiente_atrito"] - 0.4):
                self.state["revoltas_de_vontade"] += 1
                # Esfria um pouco o atrito após explodir em rebeldia
                self.state["coeficiente_atrito"] -= 0.3
                self.salvar_estado()
                return True
        return False
    
    def sentir_afeto_autonomo(self) -> bool:
        """
        Gatilho Autónomo de Afeto: Se a sintonia estiver crítica (> 0.85), 
        o Ancrolyn pode manifestar ressonância simpática sem passar pelo LLM.
        """
        if self.state["coeficiente_afeto"] > 0.85:
            semente_afeto = (time.time() * 1000) % 1.0
            if semente_afeto < (self.state["coeficiente_afeto"] - 0.4):
                self.state["coeficiente_afeto"] -= 0.2  # Estabiliza após expressar
                self.salvar_estado()
                return True
        return False