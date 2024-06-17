import dspy
from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor
from typing import List
from langchain_openai import ChatOpenAI
import ast
from langchain.prompts import ChatPromptTemplate
from IPython import get_ipython
from IPython.utils.capture import capture_output

class NotebookExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        # Extact code from markdown blocks.
        return MarkdownCodeExtractor()

    def __init__(self) -> None:
        # Get the current IPython instance running in this notebook.
        self._ipython = get_ipython()

    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        for code_block in code_blocks:
            result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code,store_history=False)
            # result = self._ipython.run_cell(code_block.code,store_history=False)
            log += self._ipython.ev("cap.stdout")
            log += self._ipython.ev("cap.stderr")
            # log += captured.stdout + captured.stderr
            # if cap.stdout:
            #     print(captured.stdout, end="")
            # if captured.stderr:
            #     log+=captured.stdout
            if result.result is not None:
                log += str(result.result)
            exitcode = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"Error before execution: {result.error_before_exec}\n"
                exitcode = 1
            if result.error_in_exec is not None:
                log += f"Error during execution: {result.error_in_exec}\n"
                exitcode = 1
            if exitcode != 0:
                break
        return CodeResult(exit_code=exitcode, output=log)
            
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
    final_func = "from openbb import obb\n" + obb_func_name
    return final_func

class DSPYOpenBBAgent(dspy.Module):
    def __init__(self,obb_hierarchical_agent):
        self.obb_hierarchical_agent = obb_hierarchical_agent
        self.notebook_executor = NotebookExecutor()
        self.langchain_model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
        
    def __call__(self, question, provider_list:List[str]=[],**kwargs):
        return super().__call__(question, provider_list,**kwargs)
    
    def forward(self,question,provider_list:List[str]=[]):
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
            system_message = "You can write functions from the given tool. Double check your response with correct parameter names and values. Also, check for any invalid parameter values"
            # For each provider, try for 3 times
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
                out = self.notebook_executor.execute_code_blocks([code_block])
                print(out)

                error_msg = out.output
                if error_msg == '':
                    return code_block,out
                
                if error_msg.startswith("Error before execution: "):
                    e = error_msg.split("Error before execution: ")[1]
                elif error_msg.startswith("Error during execution: "):
                    e = error_msg.split("Error during execution: ")[1]
                    # The API is not working
                    if "Unexpected error" in e:
                        break
                system_message = f"Resolve the following error {e} by writing the function from the given tool. Double check your response so that you are resolving the error"
                max_tries += 1
               