from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import ast

def run_function_calling(fcs,req_provider,question:str):
    for func in fcs:
        if func != []:
            for meta in func['metadatas']:
                if meta['provider_source'] == req_provider:
                    function_call = ast.literal_eval(meta['function_call'])
                    function_call['name'] = function_call['name'].rpartition('_')[0]
                    break
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human","{input}")
        ]
    )

    model = ChatOpenAI(temperature=0).bind(functions=[function_call],function_call={"name": function_call['name']})
    runnable = prompt | model
    resp = runnable.invoke({"input":question})
    return resp
