


def get_query_engine(index):
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )
    return query_engine
