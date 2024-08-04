import os
import openai as OpenAI
import torch
from langchain.chains import SequentialChain, LLMChain, SimpleSequentialChain 
from langchain_openai import ChatOpenAI
#from langchain.llms import OpenAI as LangchainOpenAI
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import whisper
# Import Prompt objects
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
# Import memory
from langchain_community.memory.kg import ConversationKGMemory
from langchain.output_parsers import CommaSeparatedListOutputParser

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model on the appropriate device
model = whisper.load_model("base", device=device)

# API-KEY
OPENAI_API_KEY = ''

# Set the API key in the system environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load Whisper model
model = whisper.load_model("base")

def audio_to_text(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]


def text_to_audio(text):
    result = OpenAI.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
        )
    return result


# Create ChatOpenAI object with API_KEY
chatgpt = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
memory = ConversationKGMemory(llm=chatgpt, output_key="instrucciones")

# Initialize comma separated parser
output_parser = CommaSeparatedListOutputParser()

# Get instructions from parser in String format
format_instructions = output_parser.get_format_instructions()


### Prompts

## system + human prompt

# Main template
template_sistema = """Eres un chef de cocina experto que da consejos breves sobre recetas de cocina con el tiempo y los ingredientes que se tienen en casa."""

# Create basic prompt with the main template
prompt_sistema = PromptTemplate(template=template_sistema)

# Create SystemMessagePromptTemplate
template_sistema = SystemMessagePromptTemplate(prompt=prompt_sistema)

# Now the same for the human with an empty template
prompt_humano = PromptTemplate(template="{pregunta}", input_variables=["pregunta"])
template_humano = HumanMessagePromptTemplate(prompt=prompt_humano)

# Chat template created from system and human prompts
chat_prompt = ChatPromptTemplate.from_messages([template_sistema, template_humano])

# Create LLMChain from chat_prompt
# Create LLMChain from chat_prompt
# chat_chain = LLMChain(llm=chatgpt, prompt=chat_prompt, output_key="platillo")
chat_chain = chat_prompt | chatgpt

## Output parser prompt

# Initialize comma separated parser
output_parser = CommaSeparatedListOutputParser()

# Get instructions from parser in String format
format_instructions = output_parser.get_format_instructions()

# Template with input variables {platillo} & {parsear}
template_ingredientes_parser = """¿Cuáles son los ingredientes y su porción para preparar: {platillo}\n{parsear}?"""

# Prompt for generating ingredients list
prompt_ingredientes_parser = PromptTemplate(template=template_ingredientes_parser, input_variables=["platillo"], partial_variables={"parsear":format_instructions + '\n Add the "platillo" name to the list'})

# Create LLMChain from prompt_ingredientes_parser
# Create LLMChain from prompt_ingredientes_parser
# ingredientes_chain = LLMChain(llm=chatgpt, prompt=prompt_ingredientes_parser, output_key="ingredientes")
ingredientes_chain = prompt_ingredientes_parser | chatgpt

## Instructions prompt

# Template with input variables {platillo} & {parsear}
template_instrucciones_parser = """¿Cuáles son los pasos para preparar: {platillo} con los siguientes ingredientes: {ingredientes}?"""

# Prompt for generating instruccions list
prompt_instrucciones_parser = PromptTemplate(template=template_instrucciones_parser, input_variables=["platillo", "ingredientes"])

# Create LLMChain from prompt_instrucciones_parser
# Create LLMChain from prompt_instrucciones_parser
#instrucciones_chain = LLMChain(llm=chatgpt, prompt=prompt_instrucciones_parser, output_key="instrucciones")
instrucciones_chain = prompt_instrucciones_parser | chatgpt

# Combine chains
combined_chain = chat_chain | ingredientes_chain | instrucciones_chain

# class ChatBot(SequentialChain):
class ChatBot:
    
    def __init__(self):

        """
        super().__init__(
            #chains=[final_chain],
            chains=[chat_chain, ingredientes_chain, instrucciones_chain],
            input_variables=["pregunta"],
            output_variables=["platillo", "ingredientes", "instrucciones"],
            verbose=True,
            memory=memory,
        )
        """
        self.chain = combined_chain
    """
    def __init__(self):
        super().__init__(
            steps=[chat_chain, ingredientes_chain, instrucciones_chain],
            memory=memory,
        )
    """

    def format_response(self, response):
        # Parse the platillo to extract the dish name
        platillo_name = response["platillo"].split('\n')[0].strip()
        
        # Ensure ingredientes is a list
        ingredientes_list = response["ingredientes"].split(", ")

        # Ensure instrucciones is a list of steps
        instrucciones_list = response["instrucciones"].split("\n")

        # Create the formatted JSON response
        formatted_response = {
            "pregunta": response["pregunta"],
            "Platillo": platillo_name,
            "ingredientes": ingredientes_list,
            "instrucciones": instrucciones_list
        }
        
        return formatted_response

    def process_audio(self, audio_path):
        # Transcribe audio to text
        print("Processing audio from path:", audio_path)
        pregunta = audio_to_text(audio_path)
        print("Transcribed text:", pregunta)
        
        # Generate the response using the transcribed text
        response = self.chain.invoke({"pregunta": pregunta})
        print("process audio ok", audio_path)
        return response
        
    def instrucciones_to_audio(self, instrucciones, output_path):
        # Join instructions into a single string
        instrucciones_text = "\n".join(instrucciones)
        # Convert text to audio and save to file
        return text_to_audio(instrucciones_text).stream_to_file(output_path)
        
"""

# Initialize the chatbot
chatbot = ChatBot()

# Process an audio file and get the formatted response
audio_path = "C:/Users/ramur/Downloads/2024-04-18 21-40-10.wav"
transcripted_audio = chatbot.process_audio(audio_path)
formatted_response = chatbot.format_response(transcripted_audio)
print(formatted_response)

# Convert instructions to audio file
instrucciones_audio_path = "C:/Users/ramur/OneDrive/Escritorio/Recipe_bot/app/uploads/prueba.wav"

print(f"Instrucciones audio saved to: {instrucciones_audio_path}")

from IPython.display import Audio
# Reproducir el archivo de audio
instrucciones_audio = chatbot.instrucciones_to_audio(formatted_response["instrucciones"], instrucciones_audio_path)
Audio(filename=str(instrucciones_audio_path), autoplay=True)
"""