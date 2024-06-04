from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import ast
from typing import List

from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor
from typing import List

from IPython import get_ipython

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
            result = self._ipython.run_cell("%%capture --no-display cap\n" + code_block.code)
            log += self._ipython.ev("cap.stdout")
            log += self._ipython.ev("cap.stderr")
            if result.result is not None:
                log += str(result.result)
            exitcode = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"\n{result.error_before_exec}"
                exitcode = 1
            if result.error_in_exec is not None:
                log += f"\n{result.error_in_exec}"
                exitcode = 1
            if exitcode != 0:
                break
        return CodeResult(exit_code=exitcode, output=log)


def format_function(function_response):
    obb_func = function_response.additional_kwargs["function_call"]
    obb_func_name = obb_func["name"]
    obb_func_name = ".".join(obb_func_name.split("-")) + "("
    args_dict = ast.literal_eval(obb_func["arguments"])
    for arg, val in args_dict.items():
        if isinstance(val,str):
            val = f"'{val}'"
        obb_func_name += f"{arg}={val},"
    obb_func_name = obb_func_name[:-1] + ")"
    final_func = "from openbb import obb\n" + obb_func_name
    return final_func

def main_function_calling(obb_chroma,question:str,provider_list:List[str]=[]):
    funcs,prompts = obb_chroma(question)
    print(funcs)
    provider_sources = [fn['provider_source'] for fn in funcs[0]['metadatas']]
    if provider_list != []:
        valid_providers = [sources for sources in provider_sources if sources in provider_list]
    else:
        valid_providers = provider_sources
    for func in funcs:
        if func != []:
            for meta in func["metadatas"]:
                if meta["provider_source"] == valid_providers[0]:
                    function_call = ast.literal_eval(meta["function_call"])
                    function_call["name"] = function_call["name"].rpartition("_")[0]
                    break
    
    prompt = ChatPromptTemplate.from_messages([("human", "{input}"),("system","You can write functions from the given tool. Double check your response with correct parameter names and values. Also, check for any invalid parameter values")])

    model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo").bind(
        functions=[function_call], function_call={"name": function_call["name"]}
    )
    runnable = prompt | model
    resp = runnable.invoke({"input": question})

    obb_func = format_function(resp)
    code_block = CodeBlock(language="python", code=obb_func)
    return code_block


def run_function(code_block):
    notebook_executor = NotebookExecutor()

    notebook_executor.execute_code_blocks([code_block])
