import networkx as nx
from typing import List
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
from chromadb.utils.batch_utils import create_batches

from dotenv import load_dotenv
def build_graph():
    with open("openbb_functions_enum.json", "r") as f:
        openbb_functions = json.load(f)
    with open("reference.json", "r") as f:
        data = json.load(f)
    openbb_functions_name_dict = {k['name']:k for k in openbb_functions}
    routers_names = [r.split("/")[1] for r in data['routers']]
    all_functions = list(openbb_functions_name_dict.keys())

    def get_graph(router_name:str):
        # print(router_name)
        desc = data['routers']['/'+router_name]

        func_name = f'obb_{router_name}'
        path_name = "/" + router_name
        req_func = [rf for rf in all_functions if rf.startswith(func_name)]
        req_data = [d for d in data['paths'] if d.startswith(path_name)]
        G = nx.DiGraph()

        G.add_nodes_from([(router_name,{"type":"level_1","description":desc['description']})])
        for rd in req_data:
            prev_node = router_name
            router_path_split = rd.split("/")[2:]
            trail = f"{router_name}"
            base_func_name = f"obb_{router_name}_" + "_".join(router_path_split)
            for idx,rps in enumerate(router_path_split):
                if idx == len(router_path_split)-1:
                    # It has multiple providers
                    if base_func_name in openbb_functions_name_dict:
                        standard_func_params = openbb_functions_name_dict[base_func_name]
                        if not G.has_node(rps):
                            G.add_nodes_from([(rps,{"type":f"level_{idx+2}","description":standard_func_params['description'],"trail":trail,'peanultimate_node':True})])
                            G.add_edge(prev_node,rps)
                        trail += f"-->{rps}"
                        all_providers = standard_func_params['parameters']['properties']['provider']['enum']
                        for provider in all_providers:
                            provider_func_name = base_func_name + "_" + provider
                            provider_func_params = openbb_functions_name_dict[provider_func_name]
                            G.add_nodes_from([(provider_func_name,{"type":f"provider_function","function_call":provider_func_params,"trail":f"{trail}","provider_source":provider})])
                            G.add_edge(rps,provider_func_name)
                    else:
                        trail += f"-->{rps}"
                        for obb_funcs in openbb_functions_name_dict:
                            if base_func_name in obb_funcs:
                                provider_func = openbb_functions_name_dict[obb_funcs]
                                if not G.has_node(rps):
                                    G.add_nodes_from([(rps,{"type":f"level_{idx+2}","description":provider_func['description'],"trail":trail,'peanultimate_node':True})])
                                    G.add_edge(prev_node,rps)
                                provider = obb_funcs.rpartition("_")[-1]
                                G.add_nodes_from([(obb_funcs,{"type":f"provider_function","function_call":provider_func,"trail":f"{trail}","provider_source":provider})])
                                G.add_edge(rps,obb_funcs)
                                break
                else:
                    if not G.has_node(rps):
                        G.add_nodes_from([(rps,{"type":f"level_{idx+2}","description":data['paths'][rd]['description'],"trail":trail,'peanultimate_node':False})])
                        G.add_edge(prev_node,rps)
                    trail += f"-->{rps}"
                    prev_node = rps
        return G
    router_names_graph = {k:get_graph(k) for k in routers_names}
    return router_names_graph

def build_docs_metadata(router_names_graph):
    embed_docs = []
    embed_metadata = []
    non_embed_docs =[]
    non_embed_metadata = []
    for router_name,router_graph in router_names_graph.items():
        for node,attributes in router_graph.nodes(data=True):
            if attributes['type'].startswith("level"):
                embed_docs.append(attributes['description'])
                attributes.update({"node_name":node})
                for key,value in attributes.items():
                    if isinstance(value,dict):
                        attributes[key] = str(value)
                    else:
                        pass
                embed_metadata.append(attributes)
            else:
                if 'description' in attributes:
                    non_embed_docs.append(attributes['description'])
                else:
                    non_embed_docs.append("empty")
                attributes.update({"node_name":node})
                for key,value in attributes.items():
                    if isinstance(value,dict):
                        attributes[key] = str(value)
                    else:
                        pass
                non_embed_metadata.append(attributes)

    docs = embed_docs + non_embed_docs
    metadata = embed_metadata + non_embed_metadata
    return docs, metadata

def build_database(docs,metadata,path="OpenBB",collection_name:str="obb"):
    load_dotenv(override=True)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model_name="text-embedding-3-small")

    client = chromadb.PersistentClient(path=path)
    if client.get_collection(name=collection_name):
        client.delete_collection(name=collection_name)

    openbb_collection = client.create_collection(name=collection_name,embedding_function=emb_fn)

    openbb_ids = [f"id{i}" for i in range(len(docs))]
    batches = create_batches(api=client,ids=openbb_ids, documents=docs, metadatas=metadata)
    for batch in batches:
        openbb_collection.add(ids=batch[0],
                    documents=batch[3],
                    metadatas=batch[2])
    return openbb_collection

def load_database(path="OpenBB",collection_name:str="obb"):
    load_dotenv(override=True)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model_name="text-embedding-3-small")
    client = chromadb.PersistentClient(path=path)
    openbb_collection = client.get_collection(name=collection_name,embedding_function=emb_fn)
    return openbb_collection

