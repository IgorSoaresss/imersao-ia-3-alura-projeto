# instalações

%pip -q install google-genai
%pip -q install google-adk


# importações genai

import os

from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

from google import genai
client = genai.Client()
MODEL_ID = "gemini-2.0-flash"

from IPython.display import HTML, Markdown


# importações adk

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from datetime import date
import textwrap
from IPython.display import display, Markdown
import requests
import warnings

warnings.filterwarnings("ignore")


def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


##########################################
# --- Agente 1: Conhecendo a pessoa --- #
##########################################

def agente_buscador(intimidade, idade, genero, gosta, naogosta, personalidade):
  buscador = Agent(
      name = "agente_buscador",
      model = "gemini-2.0-flash",
      instruction = """
      Você é um assistente de pesquisa. A sua tarefa é usar a ferramenta de
      busca do google (google_search) para ver os presentes dados mais 
      comummente para pessoas que se encaixam em cada campo fornecido. Se algum 
      campo contiver apenas a letra N, considere que ficou em branco. Cada 
      campo pode conter vícios de linguagem por serem tirados de uma conversa - 
      extraia a informação-chave obtida e simplifique para pesquisar. Gere uma 
      lista com, no máximo, 5 presentes para CADA campo fornecido (5 para 
      parentesco, 5 para intimidade, etc), da forma mais sucinta possível. Opte 
      por fontes que baseiem suas informações em argumentos e dados, mas não os 
      informe.
      """,
      description = "Agente que busca presentes no Google sobre cada campo informado",
      tools = [google_search],
  )
  entrada_do_agente_buscador = f"Intimidade: {intimidade}\nIdade: {idade}\nGênero: {genero}\nGosta de: {gosta}\nNão gosta de: {naogosta}\nPersonalidade: {personalidade}"
  presentes_buscados = call_agent(buscador, entrada_do_agente_buscador)
  return presentes_buscados


################################################
# --- Agente 2: Planejador de posts --- #
################################################
def agente_planejador(presentes_buscados):
    planejador = Agent(
        name = "agente_planejador",
        model = "gemini-2.0-flash",
        instruction = """
        Você é um estudioso da área do comportamento humano, especialista em 
        gerar feedbacks positivos. Com base na lista de presentes buscados, 
        você deve: usar a ferramenta de busca do Google (google_search) para 
        criar um plano sobre quais presentes são mais relevantes e que poderiam 
        gerar mais alegria no contexto informado. Você também pode usar o 
        google_search para encontrar mais informações sobre as opções e 
        apresentar explicações. Ao final, você irá escolher a opção mais 
        relevante entre elas com base nas suas pesquisas (se basear em dados 
        concretos) e retornar esse opção, seus pontos mais relevantes e um 
        plano com variação de preço ou outras que podem impactar no processo.
        """,
        description = "Agente que planeja posts",
        tools=[google_search]
    )

    entrada_do_agente_planejador = f"Presentes buscados: {presentes_buscados}"
    planejamento = call_agent(planejador, entrada_do_agente_planejador)
    return planejamento


######################################
# --- Agente 3: Redator do Post --- #
######################################
def agente_redator(planejamento):
    redator = Agent(
        name="agente_redator",
        model="gemini-2.0-flash",
        instruction="""
            Você é um Redator Criativo especializado em criar posts virais e 
            diálogos que roubam a atenção. Você escreve textos da forma mais 
            clara e sucinta possível, sem ocultar informações. Deixe claro a 
            opção mais recomendada de presente, e em seguida faça um discurso 
            encorajador que a pessoa que vai entregar o presente vai falar, 
            para ter um impacto ainda melhor para pessoa a ser presenteada. O 
            discurso deve ser engajador, informativo, com linguagem simples e 
            ter sua intimidade e formalidade de acordo com o que já foi 
            fornecido anteriormente.
            """,
        description="Agente redator de posts engajadores para Instagram"
    )
    entrada_do_agente_redator = f"Presentes selecionados: {planejamento}"
    presentes_selecionados = call_agent(redator, entrada_do_agente_redator)
    return presentes_selecionados


##########################################
# --- Agente 4: Revisor de Qualidade --- #
##########################################
def agente_revisor(presentes_selecionados):
    revisor = Agent(
        name="agente_revisor",
        model="gemini-2.0-flash",
        instruction="""
            Você é um Editor e Revisor de Conteúdo meticuloso, especializado em 
            diálogo digno de ser verbalizado. Você precisa deixar o texto o 
            mais respeitável possível, de forma que mesmo que possa ser 
            informal, pareça influente. Não exagere nas palavras, seja parcial, 
            e foque em corrigir eventuais erros ortográficos. O que você vai 
            receber é um rascunho, que pode estar bom ou não. Se o rascunho 
            estiver bom, responda apenas 'O rascunho está ótimo e pronto para 
            publicar!'. Caso haja problemas, aponte-os e gere a versão final.
            """,
        description="Agente revisor de post para redes sociais."
    )
    entrada_do_agente_revisor = f"Presentes selecionados: {presentes_selecionados}"
    versao_final = call_agent(revisor, entrada_do_agente_revisor)
    return versao_final


print("❓ Para maior certeza sobre o presente a ser escolhido, responda 7 perguntas, de forma curta e direta (vai valer a pena!).\nDigite N caso não queira responder alguma delas.")

nome = input("\n1 - Qual é o nome da pessoa a ser presenteada? -> ")
while not nome:
    print("Você precisa digitar o nome!")
    nome = input("❓ Por favor, digite o nome da pessoa a ser presenteada: ")
    if nome == "N":
      break

intimidade = input("2 - O quão íntimo você se considera dela? -> ")
while not intimidade:
    print("Você precisa digitar o grau de intimidade!")
    intimidade = input("❓ Por favor, digite o quão íntimo você se considera da pessoa a ser presenteada: ")
    if intimidade == "N":
      break

idade = input("3 - Qual é a idade dela? -> ")
while not idade:
    print("Você precisa digitar a idade dela!")
    idade = input("❓ Por favor, digite a idade da pessoa a ser presenteada: ")
    if idade == "N":
      break

genero = input("4 - Qual é o gênero dela? -> ")
while not genero:
    print("Você precisa digitar o gênero dela!")
    genero = input("❓ Por favor, digite o gênero da pessoa a ser presenteada: ")
    if genero == "N":
      break

gosta = input("5 - Coisas ou ações que ela gosta -> ")
while not gosta:
    print("Você precisa digitar algum gosto ou costume!")
    gosta = input("❓ Por favor, digite algum costume ou gosto especial da pessoa a ser presenteada: ")
    if gosta == "N":
      break

naogosta = input("6 - Coisas ou ações que ela NÃO gosta -> ")
while not naogosta:
    print("Você precisa digitar algum gosto ou costume!")
    gosta = input("❓ Por favor, digite algum costume ou gosto especial da pessoa a ser presenteada: ")
    if gosta == "N":
      break

personalidade = input("7 - Ela é uma pessoa mais extrovertida ou introvertida?  -> ")
while not personalidade:
    print("Você precisa digitar sobre a personalidade dela!")
    personalidade = input("❓ Por favor, digite se a pessoa é mais extrovertida ou introvertida: ")
    if personalidade == "N":
      break

if not nome:
  print("Terminamos as perguntas! Vamos, então, checar as melhores opções de presente para esta pessoa...")

else:
  print(f"Terminamos as perguntas! Vamos, então, checar as melhores opções de presente para {nome}...")

presentes_buscados = agente_buscador(intimidade, idade, genero, gosta, naogosta, personalidade)
print("\n--- 📝 Resultado do Agente 1 (Buscador) ---\n")
display(to_markdown(presentes_buscados))
print("-----------------------------------------------------------")

planejamento = agente_planejador(presentes_buscados)
print("\n--- 📝 Resultado do Agente 2 (Planejador) ---\n")
display(to_markdown(planejamento))
print("-----------------------------------------------------------")

presentes_selecionados = agente_redator(planejamento)
print("\n--- 📝 Resultado do Agente 3 (Redator) ---\n")
display(to_markdown(presentes_selecionados))
print("-----------------------------------------------------------")

versao_final = agente_revisor(presentes_selecionados)
print("\n--- 📝 Resultado do Agente 4 (Revisor) ---\n")
display(to_markdown(versao_final))
print("-----------------------------------------------------------")
