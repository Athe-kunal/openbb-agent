from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import ast


def run_function_calling(fcs, req_provider, question: str):
    for func in fcs:
        if func != []:
            for meta in func["metadatas"]:
                if meta["provider_source"] == req_provider:
                    function_call = ast.literal_eval(meta["function_call"])
                    function_call["name"] = function_call["name"].rpartition("_")[0]
                    function_call['name'] = "_".join(function_call["name"].split("-"))
                    break
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])

    model = ChatOpenAI(temperature=0).bind(
        functions=[function_call], function_call={"name": function_call["name"]}
    )
    runnable = prompt | model
    resp = runnable.invoke({"input": question})
    return resp


def format_function(function_response):
    obb_func = function_response.additional_kwargs["function_call"]
    obb_func_name = obb_func["name"]
    obb_func_name = ".".join(obb_func_name.split("-")) + "("
    args_dict = ast.literal_eval(obb_func["arguments"])
    for arg, val in args_dict.items():
        obb_func_name += f"{arg}={val},"
    obb_func_name = obb_func_name[:-1] + ")"
    final_func = "from openbb import obb\n" + obb_func_name
    return final_func
