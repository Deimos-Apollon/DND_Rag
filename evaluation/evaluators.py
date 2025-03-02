import os

from tqdm import tqdm
from llama_index.llms.gigachat import GigaChatLLM
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    RetrieverEvaluator,
    generate_question_context_pairs,
)

from src.rag_components.assistant import Assistant


assistant = Assistant()
llm = GigaChatLLM(
    credentials=os.getenv("gigachat_credentials"),
    verify_ssl_certs=False,
)


def get_faithfulness_evaluation_score(eval_queries):
    """Calculate average faithfulness score for responses to evaluation queries.
    
    Faithfulness measures whether the response is grounded in the provided context 
    and doesn't contain hallucinations.
    
    Args:
        eval_queries (list[str]): List of queries/questions to evaluate
        
    Returns:
        float: Average faithfulness score across all queries (0.0-1.0 scale)
    """
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)

    scores = []
    for eval_query in tqdm(eval_queries, desc="Faithfulness_evaluation"):
        response_vector = assistant._get_response(eval_query)
        eval_result = faithfulness_evaluator.evaluate_response(
            response=response_vector
        )
        scores.append(eval_result.score)

    return sum(scores) / len(scores)


def get_relevancy_evaluation_score(eval_queries):
    """Evaluate relevancy of responses to evaluation queries.
    
    Relevancy measures whether the response answers the query effectively using 
    relevant source context information.
    
    Args:
        eval_queries (list[str]): List of queries/questions to evaluate
        
    Returns:
        float: Average relevancy score across all queries (0.0-1.0 scale)
    """
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    eval_result = []

    for eval_query in tqdm(eval_queries, desc="Relevancy evaluation"):
        response_vector = assistant._get_response(eval_query)
        eval_source_result_full = [
            relevancy_evaluator.evaluate(
                query=eval_query,
                response=response_vector.response,
                contexts=[source_node.get_content()],
            )
            for source_node in response_vector.source_nodes
        ]
        eval_res = any(
            result.passing is not False for result in eval_source_result_full
        )
        eval_result.append(eval_res)

    return sum(eval_result) / len(eval_result)


def get_correctness_score(eval_qa):
    """Evaluate correctness of responses against reference answers.
    
    Correctness measures factual accuracy and completeness compared to 
    ground truth reference answers.
    
    Args:
        eval_qa (list[dict]): List of QA pairs in format 
            [{"question": "...", "answer": "..."}, ...]
            
    Returns:
        float: Average correctness score across all queries (0.0-1.0 scale)
    """
    correctness_evaluator = CorrectnessEvaluator(llm=llm)
    passes = []

    for qa_pair in tqdm(eval_qa, desc="Correctness evaluation"):
        eval_query = qa_pair["question"]
        reference = qa_pair["answer"]
        response = assistant._get_response(eval_query)
        correctness_result = correctness_evaluator.evaluate(
            query=eval_query,
            response=response.response,
            reference=reference,
        )
        passes.append(correctness_result.passing)

    return sum(passes) / len(passes)
