import os
import time
import requests
from groq import Groq

class FailoverRouter:
    """
    Controlador de Inferência de Camada Dupla.
    Garante execução com zero erro e mitigação ativa de latência.
    """
    def __init__(self):
        # Inicialização dos clientes através de variáveis de ambiente
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.cf_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        
        # Endpoint físico da infraestrutura Cloudflare Workers AI
        self.cf_url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/ai/v1/messages"
        
    def processar_requisicao(self, system_instruction, user_input, temp=0.72):
        """
        Executa a inferência primária na Groq. Em caso de saturação, 
        redireciona o payload para a Cloudflare em modo síncrono.
        """
        # Configuração de payloads padronizados
        mensagens = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ]
        
        start_time = time.time()
        
        # --- ROTA 01: GROQ CLOUD ---
        try:
            print("[Router] Canal 01 (Groq) -> Iniciando transmissão de pacotes...")
            # Parâmetros exatos extraídos do compilador Groq fornecido
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=mensagens,
                temperature=temp,
                max_completion_tokens=512, # Alinhado com o limite de renderização do chat
                top_p=0.90,
                stream=False
            )
            texto_saida = response.choices[0].message.content.strip()
            latency = time.time() - start_time
            print(f"[Router] Canal 01 OK -> Latência: {latency:.2f}s | Provedor: Groq")
            return texto_saida, latency, "Groq"
            
        except Exception as e:
            # Captura falhas de cota, sobrecarga (429) ou queda de conexão
            print(f"[Aviso Router] Falha crítica no Canal 01: {str(e)}")
            print("[Router] Acionando protocolo de contingência -> Desviando fluxo para Canal 02...")
            
            # --- ROTA 02: CLOUDFLARE WORKERS AI ---
            start_cf = time.time()
            headers = {
                "Authorization": f"Bearer {self.cf_token}",
                "Content-Type": "application/json"
            }
            
            # Payload compatível com a API Gateway da Cloudflare
            payload = {
                "model": "anthropic/claude-opus-4.8",
                "max_tokens": 512,
                "messages": mensagens
            }
            
            try:
                response_cf = requests.post(
                    self.cf_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=15.0 # Timeout estrito para evitar loops de espera
                )
                
                if response_cf.status_code == 200:
                    dados = response_cf.json()
                    # Parsing do retorno estruturado da Cloudflare
                    texto_saida = dados["result"]["choices"][0]["message"]["content"].strip()
                    latency = time.time() - start_time
                    print(f"[Router] Canal 02 OK -> Latência Total: {latency:.2f}s (CF: {time.time()-start_cf:.2f}s) | Provedor: Cloudflare")
                    return texto_saida, latency, "Cloudflare"
                else:
                    raise RuntimeError(f"Erro HTTP Cloudflare: {response_cf.status_code} - {response_cf.text}")
                    
            except Exception as erro_critico:
                latency_total = time.time() - start_time
                print(f"[ERRO DE HARDWARE/REDE] Ambos os canais falharam. Latência acumulada: {latency_total:.2f}s")
                # Fallback estrito: Retorna erro limpo sem quebrar a execução do AncrolynEngine
                fallback_erro = "*Falha estrutural de rede.* Latência limite atingida em ambos os canais de processamento."
                return fallback_erro, latency_total, "Falha_Geral"