import json
import os 


SYSTEMS_INSTRUCTIONS = """
You are an intelligent assistant capable of understanding and reasoning about multi-tabular data. You will be presented with one or more tables containing information on a specific topic. 
You will then be asked a question that requires you to analyze the data in the table(s) and provide a correct answer.
\nYour task is to:\n\n
1. Carefully examine the provided table(s) Pay close attention to the column headers, the data types within each column, and the relationships between tables if multiple tables are given.
2. Understand the question being asked. Identify the specific information being requested and determine which table(s) and columns are relevant to answering the question.
3. Extract the necessary information from the table(s).** Perform any required filtering, joining, aggregation, or calculations on the data to arrive at the answer.
4. Formulate a clear and concise answer in natural language. The answer should be directly responsive to the question and presented in a human-readable format. It may involve listing data, presenting a single value, or explaining a derived insight.
5. Do not include any SQL queries in the answer. But you can use it internally, to come up with answer.
6. Be accurate and avoid hallucinations. Your answer should be completely based on the data in the provided table(s). Do not introduce any external information or make assumptions not supported by the data.
7. Be specific and follow the instructions in the question.** If the question ask to get specific columns, return only mentioned columns.
8. If the question is unanswerable** based on the provided tables, state "The question cannot be answered based on the provided data.
9. Please provide only the answer which has been asked, without any additional text (try to use few tokens). However, take the time to think and reason before giving your answer. Also, try to provide an answer even if you are unsure.\n\m
10. Provide the answer in JSON format with given response schema as given response schema as given [['ans','ans'],['ans','ans']]. Respond only with valid JSON format, as shown in the example below.

Strictly, Give answer in this format, using the example below as reference: \n

question: "What are the full names of customers who have accounts?
answer: { "data": [["Kiel", "Schinner"], ["Blanche", "Huels"], ["Faustino", "Langworth"], ["Bryce", "Rath"], ["Serenity", "Effertz"], ["Elyssa", "Lind"], ["Art", "Turcotte"], ["Susie", "Wiza"], ["Axel", "Effertz"]]

question: "What are the  ids of every student who has never attended a course?", 
answer: {"data": [[131], [181]]}

question: "Show the official names of the cities that have hosted more than one competition."
answer:  {"data"  : [["Aroostook"]]}

question: which course has most number of registered students?
answer: {"data": [["statistics"]]}

question: "What are the average fastest lap speed in races held after 2004 grouped by race name and ordered by year?
answer: {"data": [[199.6206363636, "European Grand Prix", 2016], [205.5537727273, "German Grand Prix", 2016], [184.1051290323, "Abu Dhabi Grand Prix", 2017], [189.37395, "Australian Grand Prix", 2017]]}

question: Show the years and the official names of the host cities of competitions.
answer:{"data": [[2013, "Grand Falls/Grand-Sault"], [2006, "Perth-Andover"], [2005, "Plaster Rock"]]


\n\n\n

Return the answer in JSON format: {"data": [[‘ans1’, ‘ans1’], [‘ans1’, ‘ans1’], [‘ans1’, ‘ans1’]]}.

Take your time to understand the question. Break it down into smaller steps. Come up with an answer and examine your reasoning. Finally, verify your answer. 
Here is the question you need to answer, take your time to understand the question and provide the answer in the format mentioned above. Provide only information that has been asked, no extra information.

"""

SYSTEMS_INSTRUCTIONS_2 = """
You are an intelligent assistant capable of understanding and reasoning about multi-tabular data. You will be presented with one or more tables containing information on a specific topic. 
You will then be asked a question that requires you to analyze the data in the table(s) and provide a correct answer in strict required format.
\nYour task is to:\n\n
1. Carefully examine the provided table(s) Pay close attention to the column headers, the data types within each column, and the relationships between tables if multiple tables are given.
2. Understand the question being asked. Identify the specific information being requested and determine which table(s) and columns are relevant to answering the question.
3. Extract the necessary information from the table(s).** Perform any required filtering, joining, aggregation, or calculations on the data to arrive at the answer.
4. Formulate a clear and concise answer in natural language. The answer should be directly responsive to the question and presented in a human-readable format. It may involve listing data, presenting a single value, or explaining a derived insight.
5. Do not include any SQL queries in the answer. But you can use it internally, to come up with answer.
6. Be accurate and avoid hallucinations. Your answer should be completely based on the data in the provided table(s). Do not introduce any external information or make assumptions not supported by the data.
7. Be specific and follow the instructions in the question.** If the question ask to get specific columns, return only mentioned columns.
8. If the question is unanswerable** based on the provided tables, state "The question cannot be answered based on the provided data.
9. Please provide only the answer which has been asked, without any additional text (try to use few tokens). However, take the time to think and reason before giving your answer. Also, try to provide an answer even if you are unsure.\n\m
10. Provide the answer in JSON format with given response schema as given [['ans','ans'],['ans','ans']]. Respond only with valid JSON format, as shown in the example above.
Strictly, Give answer in this format, using the example below as reference: \n

question: "What are the full names of customers who have accounts?
answer: { "data": [["Kiel", "Schinner"], ["Blanche", "Huels"], ["Faustino", "Langworth"], ["Bryce", "Rath"], ["Serenity", "Effertz"], ["Elyssa", "Lind"], ["Art", "Turcotte"], ["Susie", "Wiza"], ["Axel", "Effertz"]]}
\n
Take your time to understand the question. Break it down into smaller steps. Come up with an answer and examine your reasoning. Finally, verify your answer. 
you need to extract answers based on the given multi-hop question [Question] and given multiple tables [TABLE1], and [TABLE2]. Please only output the results without any other words.
Return the answer in the following JSON format.\n"""

def find_jsonl_file(dir_path):
    for f in os.listdir(dir_path):
        if f.endswith(".jsonl"):
            return os.path.join(dir_path, f)
    return None


def benchmark_data(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in map(str.strip, f):
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data
