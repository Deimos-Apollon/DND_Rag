import json
from evaluation.evaluators import get_faithfulness_evaluation_score, get_relevancy_evaluation_score, get_correctness_score

with open('evaluation/eval_data.json') as f:
    data = json.load(f)
queries = []
for elem in data:
    queries.append(elem['question'])

print("Faithfulness score:", get_faithfulness_evaluation_score(queries))
print("Relevancy score:", get_relevancy_evaluation_score(queries))
print("Correctness score:", get_correctness_score(data))
