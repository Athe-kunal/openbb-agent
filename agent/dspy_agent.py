import dspy
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import chromadb.utils.embedding_functions as embedding_functions
import os
from agent.utils import generate_pairs, generate_pairs_recursive, get_trail_list_pairs
from dotenv import load_dotenv

load_dotenv(override=True)

emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
)
first_level_llm = dspy.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=1024)
function_calling_llm = dspy.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=1024)


class FirstSecondLevel(dspy.Signature):
    """You are given a list of keys and values separated by semicolon.
    Based on the query, you have to output the key that is most relevant to the question.
    Be precise and output only the relevant key or keys.
    Don't include any other information
    """

    query = dspy.InputField(prefix="Query which you need to classify: ", format=str)
    keys_values = dspy.InputField(prefix="Keys and Values: ", format=str)
    output = dspy.OutputField(
        prefix="Relevant Key(s): ", format=str, desc="relevant keys separated by ;"
    )


dspy.settings.configure(lm=first_level_llm)


class OpenBBAgentChroma(dspy.Module):
    """OpenBB Agent for function calling"""

    def __init__(self, collection):
        """Init function for OpenBB agent"""
        super(OpenBBAgentChroma, self).__init__()
        self.collection = collection
        get_first_level = self.collection.get(where={"type": "level_1"})
        self.first_level = ""
        for first_level_metadata in get_first_level["metadatas"]:

            self.first_level += f"{first_level_metadata['node_name']}: {first_level_metadata['description']}\n"
        self.firstSecondLevel = dspy.ChainOfThought(FirstSecondLevel)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, query: str):
        function_calls_list = []
        question_emb = emb_fn([query])[0]
        first_level_answer = self.firstSecondLevel(
            query=query, keys_values=self.first_level
        ).output
        print(f"\033[92mFirst level answer: {first_level_answer}\033[0m")
        if ";" in first_level_answer:
            # ['crypto','index']
            trail_list = [[fla.strip() for fla in first_level_answer.split(";")]]

        else:
            trail_list = [[first_level_answer]]
        curr_level = 2
        while True:
            # if curr_level>3: break
            trail_list_pairs = generate_pairs_recursive(trail_list)
            print(f"\033[93mCurrent Trail: {trail_list_pairs} and level: {curr_level}\033[0m")

            trail_where_clause = get_trail_list_pairs(trail_list_pairs)
            subsequent_level = self.collection.query(
                query_embeddings=question_emb,
                where={
                    "$and": [
                        trail_where_clause,
                        {"type": {"$eq": f"level_{curr_level}"}},
                    ]
                },
                n_results=5,
            )
            # If subsequent level metadata has only element
            if len(subsequent_level["metadatas"][0]) == 1 or curr_level > 3:
                if curr_level > 3:
                    if len(function_calls_list) == 0:
                        function_calls_list.append(
                            subsequent_level["metadatas"]["function_call"]
                        )
                    return function_calls_list
                curr_trail = f"{subsequent_level['metadatas'][0][0]['trail']}-->{subsequent_level['metadatas'][0][0]['node_name']}"
                # with peanultimate node as True
                # If peanultimate node is False, then loop again
                if subsequent_level["metadatas"][0][0]["peanultimate_node"] == True:
                    function_call = self.collection.get(
                        where={
                            "$and": [
                                {"type": {"$eq": "provider_function"}},
                                {"trail": {"$eq": curr_trail}},
                            ]
                        }
                    )
                    function_calls_list.append(function_call)
                    return function_calls_list
                else:
                    trail_list.append(
                        [subsequent_level["metadatas"][0][0]["node_name"]]
                    )
                    curr_level += 1
            elif len(subsequent_level["metadatas"][0]) > 1:
                curr_trail_list = []
                subsequent_level_str = ""
                peanultimate_node_dict = {}
                for subsequent_level_metadata in subsequent_level["metadatas"][0]:
                    if subsequent_level_metadata["peanultimate_node"]:
                        function_call = self.collection.get(
                            where={
                                "$and": [
                                    {"type": {"$eq": "provider_function"}},
                                    {
                                        "trail": {
                                            "$eq": f"{subsequent_level_metadata['trail']}-->{subsequent_level_metadata['node_name']}"
                                        }
                                    },
                                ]
                            }
                        )
                        peanultimate_node_dict.update(
                            {subsequent_level_metadata["node_name"]: function_call}
                        )
                        if curr_trail_list == []:
                            curr_trail_list.append(
                                [subsequent_level_metadata["node_name"]]
                            )
                        else:
                            curr_trail_list[-1].append(
                                subsequent_level_metadata["node_name"]
                            )
                    subsequent_level_data = (
                        subsequent_level_metadata["description"]
                        .replace("\n\n", "")
                        .replace("\n", "")
                    )
                    subsequent_level_str += f"{subsequent_level_metadata['node_name']}: {subsequent_level_data}\n\n"
                print(
                    f"\033[91mSubsequent level {curr_level} string to LLM: {subsequent_level_str}\033[0m"
                )
                if subsequent_level_str != "":
                    subsequent_level_answer = self.firstSecondLevel(
                        query=query, keys_values=subsequent_level_str
                    )
                    print(f"\033[94mLLM Answer: {subsequent_level_answer}\033[0m", )
                    splitted_subsequent_level_answer = (
                        subsequent_level_answer.output.split(";")
                    )
                    if curr_trail_list == []:
                        curr_trail_list.append(
                            [sl.strip() for sl in splitted_subsequent_level_answer]
                        )
                    else:
                        curr_trail_list[-1].extend(
                            [sl.strip() for sl in splitted_subsequent_level_answer]
                        )
                for node_name in peanultimate_node_dict:
                    function_val = peanultimate_node_dict[node_name]
                    if node_name in splitted_subsequent_level_answer:
                        if function_val != []:
                            function_calls_list.append(
                                peanultimate_node_dict[node_name]
                            )
                    else:
                        curr_trail_list[-1].remove(node_name)
                curr_trail_list[-1] = list(set(curr_trail_list[-1]))
                trail_list.extend(curr_trail_list)
                curr_level += 1
            else:
                break
        return function_calls_list


class OpenBBAgentBM25(dspy.Module):
    """OpenBB Agent for function calling"""

    def __init__(self, collection):
        """Init function for OpenBB agent"""
        super(OpenBBAgentBM25, self).__init__()
        self.collection = collection
        get_first_level = self.collection.get(where={"type": "level_1"})
        self.first_level = ""
        for first_level_metadata in get_first_level["metadatas"]:

            self.first_level += f"{first_level_metadata['node_name']}: {first_level_metadata['description']}\n"
        self.firstSecondLevel = dspy.ChainOfThought(FirstSecondLevel)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def BM25RetrieverLangchain(
        self, question: str, trail_where_clause, curr_level: int
    ):
        if curr_level > 3:
            vectordb_docs = self.collection.get(
                where={
                    "$and": [trail_where_clause, {"type": {"$eq": "provider_function"}}]
                }
            )
            langchain_docs = []
            if len(vectordb_docs["metadatas"]) == 0:
                return [Document(page_content="")]
            for data in vectordb_docs["metadatas"]:
                langchain_docs.append(Document(page_content="empty", metadata=data))
        else:
            vectordb_docs = self.collection.get(
                where={
                    "$and": [
                        trail_where_clause,
                        {"type": {"$eq": f"level_{curr_level}"}},
                    ]
                }
            )
            langchain_docs = []
            if len(vectordb_docs["metadatas"]) == 0:
                return [Document(page_content="")]
            for data in vectordb_docs["metadatas"]:
                langchain_docs.append(
                    Document(page_content=data["description"], metadata=data)
                )
        # k_value = max(1,len(vectordb_docs['metadatas'])//2)
        bm25_retriever = BM25Retriever.from_documents(
            langchain_docs, k=5, preprocess_func=(lambda x: x.lower())
        )
        bm25_docs = bm25_retriever.invoke(question.lower())
        return bm25_docs

    def forward(self, query: str):
        function_calls_list = []
        first_level_answer = self.firstSecondLevel(
            query=query, keys_values=self.first_level
        ).output
        print(f"\033[92mFirst level answer: {first_level_answer}\033[0m")
        if ";" in first_level_answer:
            # ['crypto','index']
            trail_list = [[fla.strip() for fla in first_level_answer.split(";")]]

        else:
            trail_list = [[first_level_answer]]
        curr_level = 2
        while True:
            # if curr_level>3: break
            trail_list_pairs = generate_pairs_recursive(trail_list)
            print(f"\033[93Current Trail: {trail_list_pairs} and level: {curr_level}\033[0m")

            trail_where_clause = get_trail_list_pairs(trail_list_pairs)
            bm25_docs = self.BM25RetrieverLangchain(
                question=query,
                trail_where_clause=trail_where_clause,
                curr_level=curr_level,
            )
            # If subsequent level metadata has only element
            if len(bm25_docs) == 1 or curr_level > 3:
                doc_metadata = bm25_docs[0].metadata
                if curr_level > 3:
                    if len(function_calls_list) == 0:
                        function_calls_list.append(doc_metadata)
                    return function_calls_list
                if doc_metadata == {}:
                    break
                curr_trail = f"{doc_metadata['trail']}-->{doc_metadata['node_name']}"
                # with peanultimate node as True
                # If peanultimate node is False, then loop again
                if doc_metadata["peanultimate_node"] == True:
                    function_call = self.collection.get(
                        where={
                            "$and": [
                                {"type": {"$eq": "provider_function"}},
                                {"trail": {"$eq": curr_trail}},
                            ]
                        }
                    )
                    function_calls_list.append(function_call["metadatas"])
                    return function_calls_list
                else:
                    trail_list.append([doc_metadata["node_name"]])
                    curr_level += 1
            elif len(bm25_docs) > 1:
                curr_trail_list = []
                subsequent_level_str = ""
                peanultimate_node_dict = {}
                for subsequent_level_docs in bm25_docs:
                    subsequent_level_metadata = subsequent_level_docs.metadata
                    if subsequent_level_metadata["peanultimate_node"]:
                        function_call = self.collection.get(
                            where={
                                "$and": [
                                    {"type": {"$eq": "provider_function"}},
                                    {
                                        "trail": {
                                            "$eq": f"{subsequent_level_metadata['trail']}-->{subsequent_level_metadata['node_name']}"
                                        }
                                    },
                                ]
                            }
                        )
                        # if function_call['metadatas'] != []:
                        peanultimate_node_dict.update(
                            {
                                subsequent_level_metadata["node_name"]: function_call[
                                    "metadatas"
                                ]
                            }
                        )
                        if curr_trail_list == []:
                            curr_trail_list.append(
                                [subsequent_level_metadata["node_name"]]
                            )
                        else:
                            curr_trail_list[-1].append(
                                subsequent_level_metadata["node_name"]
                            )
                    subsequent_level_data = subsequent_level_metadata["description"]
                    subsequent_level_str += f"{subsequent_level_metadata['node_name']}: {subsequent_level_data}\n\n"
                    print(
                        f"\033[93mSubsequent level {curr_level} string to LLM: {subsequent_level_str}\033[0m"
                    )
                if subsequent_level_str != "":
                    subsequent_level_answer = self.firstSecondLevel(
                        query=query, keys_values=subsequent_level_str
                    )
                    splitted_subsequent_level_answer = (
                        subsequent_level_answer.output.split(";")
                    )
                    print(f"\033[94mLLM Answer: {subsequent_level_answer}\033[0m")
                    if curr_trail_list == []:
                        curr_trail_list.append(
                            [sl.strip() for sl in splitted_subsequent_level_answer]
                        )
                    else:
                        curr_trail_list[-1].extend(
                            [sl.strip() for sl in splitted_subsequent_level_answer]
                        )
                for node_name in peanultimate_node_dict:
                    function_val = peanultimate_node_dict[node_name]
                    if node_name in splitted_subsequent_level_answer:
                        if function_val != []:
                            function_calls_list.append(
                                peanultimate_node_dict[node_name]
                            )
                    else:
                        curr_trail_list[-1].remove(node_name)
                curr_trail_list[-1] = list(set(curr_trail_list[-1]))
                trail_list.extend(curr_trail_list)
                curr_level += 1
            else:
                break
        return function_calls_list
