import dspy
from typing import List
from langchain_openai import ChatOpenAI
import ast
from langchain.prompts import ChatPromptTemplate
import logging
from agent.notebook_executor import NotebookExecutor
from autogen.coding import CodeBlock

PYTHON_CODE = "from openbb import obb\n"+\
                "df={obb_func_name}\n" + \
                "print(df.tail(100).to_markdown(index=False))" 

            
def format_function(function_response):
    obb_func = function_response.additional_kwargs["function_call"]
    obb_func_name = obb_func["name"]
    obb_func_name = ".".join(obb_func_name.split("-")) + "("
    func_args = obb_func['arguments'].replace('null','None')
    args_dict = ast.literal_eval(func_args)
    for arg, val in args_dict.items():
        if isinstance(val,str):
            val = f"'{val}'"
        obb_func_name += f"{arg}={val},"
    obb_func_name = obb_func_name[:-1] + ")"
    final_func = PYTHON_CODE.format(obb_func_name=obb_func_name)
    return final_func

class DSPYOpenBBAgent(dspy.Module):
    def __init__(self,obb_hierarchical_agent):
        self.obb_hierarchical_agent = obb_hierarchical_agent
        self.notebook_executor = NotebookExecutor()
        self.langchain_model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="conversation.log",filemode="a",level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',force=True)
    def __call__(self, question:str, provider_list:List[str]=[],**kwargs):
        return super().__call__(question, provider_list,**kwargs)
    
    def forward(self,question:str,provider_list:List[str]=[]):
        funcs,_ = self.obb_hierarchical_agent(question)
        print(funcs)
        provider_sources = [fn['provider_source'] for fn in funcs[0]['metadatas']]
        if provider_list != []:
            valid_providers = [sources for sources in provider_sources if sources in provider_list]
        else:
            valid_providers = provider_sources
        
        for vp in valid_providers:
            for func in funcs:
                if func != []:
                    for meta in func["metadatas"]:
                        if meta["provider_source"] == vp:
                            function_call = ast.literal_eval(meta["function_call"])
                            function_call["name"] = function_call["name"].rpartition("_")[0]
                            break
            max_tries = 0
            system_message = "You can write functions from the given tool. Double check your response with correct parameter names and values.\nAlso, check for any invalid parameter values"
            # For each provider, try for 3 times to fix the error
            e = ""
            code_block = CodeBlock(code="",language="python")
            while True:
                if max_tries>3: 
                    print(f"\033[31mCouldn't resolve the error {e} with the code {code_block.code}\033[0m")
                    break
                model = self.langchain_model.bind(
                    functions=[function_call], function_call={"name": function_call["name"]}
                )
                prompt = ChatPromptTemplate.from_messages([("human", "{input}"),("system",system_message)])
                runnable = prompt | model   
                resp = runnable.invoke({"input": question})

                obb_func = format_function(resp)
                code_block = CodeBlock(language="python", code=obb_func)
                print(code_block)                  
                notebook_output = self.notebook_executor.execute_code_blocks([code_block])
                print(notebook_output)

                error_msg = notebook_output.output                  
                # API error message
                if error_msg.startswith("Error before execution: "):
                    e = error_msg.split("Error before execution: ")[1]
                # Run time error message
                elif error_msg.startswith("Error during execution: "):
                    e = error_msg.split("Error during execution: ")[1]
                    # The API is not working
                    if "Unexpected error" in e:
                        break
                # The code worked successfully
                else:
                    self.logger.info(f"{question}\nCode:\n{code_block.code}\n{notebook_output.output}")
                    return notebook_output.output
                system_message = f"Resolve the following error {e} by writing the function from the given tool and modify the current code {code_block.code}. Double check your response so that you are resolving the error"
                max_tries += 1
               