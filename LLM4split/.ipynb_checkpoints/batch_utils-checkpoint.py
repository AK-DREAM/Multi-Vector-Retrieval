import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
# 运行此API配置，需要将目录中的.env中api_key替换为自己的
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import collections
from langchain.schema import HumanMessage, SystemMessage


chat = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key="sk-7cf97c394fe34640a8872d9c4132a6c7", 
        openai_api_base='https://api.deepseek.com/v1', 
        temperature=0, 
        max_tokens=512
        )


# 定义预测函数
def predict(params):
    query, input = params
    res = chat.invoke(input)
    res = res.content

    return query, res


def gen_prompt(instruction, question, dataset_name="multiqa"):
    prompt = []
    prompt.append({"role": "system", "content": instruction})

    if dataset_name == "multiqa":  # 分解提示
        prompt.append(
            {
                "role": "system",
                "content": "Instruction: For the query below, split it into semantically aligned sub-queries, separated by |, and only output the sub-queries. Do not include any other information and explanation.",
            }
        )
        prompt.append(
            {"role": "user", "content": "What color is the Santa Anita Park logo?"}
        )
        prompt.append({"role": "assistant", "content": "Santa Anita Park| logo"})

    prompt.append({"role": "user", "content": question})
    return prompt




def call_llm(query_list, chats):
    ans = []
    with ProcessPoolExecutor(max_workers=10) as executor:

        futures = []
        for idx in range(len(query_list)):
            query = query_list[idx]
            prompt = chats[idx]
            job = executor.submit(predict, (query, prompt))
            futures.append(job)


        query2res = collections.defaultdict(int) # 因为异步等待结果，返回的顺序是不定的，所以记录一下进程和输入数据的对应
        # 异步等待结果（返回顺序和原数据顺序可能不一致） ，直接predict函数里返回结果？
        for job in as_completed(futures):
            query, res = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
            query2res[query] = res
            
    for idx in range(len(query_list)):
        ans.append(
            query2res[query_list[idx]]
        )
    
    return ans