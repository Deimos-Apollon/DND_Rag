import json
from evaluation.evaluators import get_faithfulness_evaluation_score, get_relevancy_evaluation_score, get_correctness_score

with open('evaluation/eval_data.json') as f:
    data = json.load(f)
queries = []
for elem in data:
    queries.append(elem['question'])

#print("Faithfulness score:", get_faithfulness_evaluation_score(queries))
#print("Faithfulness score:", get_relevancy_evaluation_score(queries))
#print("Faithfulness score:", get_correctness_score(data))

list_relev = [True, True, True, True, False, True, True, False, True, False, True, False, True, False, True, True, True, False, False, True, False, True, False, True, True, False, False, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, False, True, True, True, False, True, True, True, False, True]
list_scores = [True, True, True, True, True, True, True, True, True, False, True, False, True, False, True, False, True, True, False, False, True, True, True, False, False, True, True, True, True, False, True, True, False, True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, True, True, False, False, True, True, False, True, True, True, True, True]
print(sum(list_relev)/ len(list_relev))
print(sum(list_scores)/ len(list_scores))