from src.evaluator.eval_retriever.evaluator import RetieverEvaluator

def evaluate(retriever, top_k, query_df, return_miss = False, evidence_column = 'similar_document',  **retrieve_params):
    retrieved_texts_list = []
    for query in query_df['problem'].values:
        retrieved_texts = retriever.retrieve(query, top_k=top_k, **retrieve_params)
        retrieved_texts_list.append(retrieved_texts[evidence_column])
    evaluator = RetieverEvaluator(query_df['evidence'], retrieved_texts_list)
    recall_score = evaluator.compute_mean_recall()

    if return_miss:
        return recall_score, evaluator.failed_retrieved_evidence

    return recall_score