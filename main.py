"""

This is the main function.
It defines the model, the tools available to the agent, and handles input/output

"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv  
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
import tools

# import tools
get_weather = tools.get_weather
search_documents = tools.search_documents
get_full_document = tools.get_full_document

# define model
# https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/?_gl=1*1c0k3xw*_gcl_au*NTc0OTgxNTExLjE3NjQ3NjAwNDk.*_ga*MTE5ODEzOTcyNS4xNzY0NzYwMDQ5*_ga_47WX3HKKY2*czE3NjQ3NjAwNDkkbzEkZzEkdDE3NjQ3NjEyNTYkajYwJGwwJGgw#langchain_openai.chat_models.ChatOpenAI
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

# create agent
agent = create_agent(
    model=model,
    tools=[get_weather, search_documents, get_full_document], 
    system_prompt="You are a helpful assistant",
)

# run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "What is happening in Algeria right now? After can you return the entire document we have on Algeria."}]}
)

print(res["messages"][-1].content)