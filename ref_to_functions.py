import re
import json
import re
import json
from copy import deepcopy

def process_params(params_desc):
            
    if "type" in params_desc:
        if "None" in params_desc['type'] and "Literal" in params_desc['type']:
            result = re.findall(r"Literal\[(.*?)\]", params_desc['type'])[0].split(", ")
            result = [value.strip('"') if value.strip() != "None" else None for value in result]    
            result = [value.replace("'", '') for value in result if value is not None]
            del params_desc['type']
            # for res in result:
                # if res in function_options:
            #         params_desc['optional'] = False
            #         break
            params_desc.update({"type":"string","enum":result})
        elif "None" not in params_desc['type'] and "Literal" in params_desc['type']:
            result = re.findall(r"Literal\[(.*?)\]", params_desc['type'])[0].split(", ")
            result = [value.replace("'", '') for value in result] 
            # for res in result:
            #     if res in function_options:
            #         params_desc['optional'] = False
            #         break
            del params_desc['type']
            params_desc.update({"type":"string","enum":result})
        elif "int" in params_desc['type']:
            params_desc['type'] = "integer"
        elif ("Union" in params_desc['type'] or "List" in params_desc['type']) and "str" in params_desc['type']:
            params_desc['type'] = "string"
        elif "str" in params_desc['type']:
            params_desc['type'] = "string"
    return params_desc
def get_curr_func(data,paths,ignore_non_standard:bool=False,params_dict={}):
    curr_func = {}
    func_name = paths.split("/")[1:]
    func_name = "obb_" + "_".join(func_name)
    curr_func['name'] = func_name
    curr_func['description'] = data["paths"][paths]['description']
    # curr_func['examples'] = data["paths"][paths]['examples']
    curr_func['parameters'] = {"type":"object","properties":{},"required":[]}
    required_params = []
    # print(paths)
    # Providers
    for params in params_dict:
    # params can be standard or other options
        if ignore_non_standard and params!="standard": break
        for params_desc in params_dict[params]:
            prop_name = params_desc['name']
            params_desc = process_params(params_desc)
            curr_func['parameters']['properties'][prop_name] = params_desc
            if "optional" in params_desc:
                if not params_desc["optional"] and prop_name not in required_params:
                    required_params.append(prop_name)
                del params_desc["optional"]
            del params_desc['name']
    curr_func['parameters']['required'] = required_params + ['provider']
    return curr_func

if __name__ == '__main__':
    with open("reference.json","r") as file:
        data = json.load(file)
    openbb_functions_enum = []
    funcs = 0
    for paths in data['paths']:
        if data['paths'][paths]['deprecated']['flag']: continue
        params_dict = data["paths"][paths]['parameters']
        function_options = list(params_dict.keys())
        if len(function_options) == 2 and function_options[0] == "standard":
            curr_func = get_curr_func(data,paths,params_dict)
            curr_func['name'] += f"_{function_options[1]}"
            openbb_functions_enum.append(curr_func)
            funcs+=len(function_options)
        elif len(function_options)>2:
            # standard function extraction
            standard_func = get_curr_func(data,paths,ignore_non_standard=True,params_dict=params_dict)
            openbb_functions_enum.append(standard_func)
            for fo in function_options[1:]:
                curr_func = deepcopy(standard_func)
                curr_func['name'] = curr_func['name'] + "_" + fo
                curr_func['description'] = f"{curr_func['description']} Get it from provider {fo}"
                provider_extra_params = params_dict[fo]
                # print(provider_extra_params)
                for params_desc in provider_extra_params:
                    prop_name = params_desc['name']
                    params_desc = process_params(params_desc)
                    curr_func['parameters']['properties'][prop_name] = params_desc
                curr_func['parameters']['properties']['provider']['enum'] = [fo]
                # curr_func['parameters']['required']
                openbb_functions_enum.append(curr_func)
            funcs+=len(function_options)
    
    json_string = json.dumps(openbb_functions_enum)

    # Specify the file path where you want to save the JSON data
    file_path = "openbb_functions_enum.json"

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write the JSON string to the file
        file.write(json_string)