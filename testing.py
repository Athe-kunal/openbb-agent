import time 
from tqdm import tqdm
import pandas as pd
import os
from agent.database import load_database
from dotenv import load_dotenv,find_dotenv
import os
from agent.dspy_agent import OpenBBAgentChroma

load_dotenv(find_dotenv(),override=True)
openbb_collection = load_database(os.environ['OPENAI_API_KEY'])

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

testing_df = pd.read_csv("openbb_question.csv")
columns = ["QUESTION","TIME","ANSWER","LLM_ANSWER","COMPLETION_TOKENS","PROMPT_TOKENS","TOTAL_TOKENS"]
if not os.path.exists(f"hierarchical_answersv2.csv"):
    df = pd.DataFrame(columns=columns,index=None)
    df.to_csv(f"hierarchical_answersv2.csv",index=False,header=True)
else:
    df = pd.read_csv(f"hierarchical_answersv2.csv",index_col=False)

pbar = tqdm(total=len(testing_df),desc="Hierarchical Answers LLM")
if not os.path.exists(f"done.txt"):
    os.mknod(f"done.txt")
done_idxs = 0
for row in testing_df.iterrows():
    obb_chroma = OpenBBAgentChroma(openbb_collection)
    if not is_file_empty(f"done.txt"):
        with open(f"done.txt","r") as f:
            done_pids = [int(x) for x in f.read().splitlines()]
        # print(done_pids)
        if done_idxs in done_pids: 
            print(f"Already done for {done_idxs}")
            pbar.update(1)
            done_idxs += 1
            continue
    path = row[1]['PATHS']
    path = path.replace("\n","").split("/")
    path[0] = "obb"
    answer = "_".join(path)

    question  = row[1]['QUESTION']
    try:
        start = time.time()
        functions,prompts = obb_chroma(question)
        end = time.time()
        time_taken = end - start

        llm_answers = []
        for fn in functions[0]['metadatas']:
            llm_answers.append(fn['node_name'].rpartition('_')[0])
        llm_answers = list(set(llm_answers))
        llm_answer = ", ".join(llm_answers)
        with open(f"done.txt","a") as f:
            f.write(f"{done_idxs}\n")
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        for prompt in prompts:
            # {'completion_tokens': 35, 'prompt_tokens': 298, 'total_tokens': 333}
            completion_tokens += prompt[0]['response']['usage']['completion_tokens']
            prompt_tokens += prompt[0]['response']['usage']['prompt_tokens']
            total_tokens += prompt[0]['response']['usage']['total_tokens']
        done_idxs+=1
    except Exception as e:
        print(e)
        llm_answer = ""
        time_taken = 0
        done_idxs+=1
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0

    df_dict = {
        "QUESTION":[question],
        "TIME":[time_taken],
        "ANSWER":[answer],
        "LLM_ANSWER":[llm_answer],
        "COMPLETION_TOKENS":[completion_tokens],
        "PROMPT_TOKENS":[prompt_tokens],
        "TOTAL_TOKENS":[total_tokens]
    }
    print(df_dict)
    curr_df = pd.DataFrame(df_dict)
    curr_df.to_csv(f"hierarchical_answersv2.csv", mode='a',index=False,header=False)
    pbar.update(1)
    if done_idxs%2 == 0:
        print(f"Sleeping for 60 seconds")
        time.sleep(60)