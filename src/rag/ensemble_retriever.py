class EnsembleRetriever:
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights if weights else [1.0] * len(retrievers)
        
    def invoke(self, query, k=3):
        all_docs = []
        for i, retriever in enumerate(self.retrievers):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                doc.metadata['retriever_rank'] = rank
                doc.metadata['retriever_weight'] = self.weights[i]
            all_docs.extend(docs)
        
        scored_docs = {}
        c = 60
        
        for doc in all_docs:
            content = doc.page_content
            rank = doc.metadata['retriever_rank']
            weight = doc.metadata['retriever_weight']
            rrf_score = weight / (rank + c)
            
            if content in scored_docs:
                scored_docs[content]['score'] += rrf_score
                if len(doc.metadata) > len(scored_docs[content]['doc'].metadata):
                    scored_docs[content]['doc'] = doc
            else:
                scored_docs[content] = {'score': rrf_score, 'doc': doc}
        
        sorted_docs = sorted(scored_docs.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:k]]