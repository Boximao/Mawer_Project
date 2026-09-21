ALLOWED_TOOLS = ("search_chunks", "get_chunk", "get_document_meta")


class ToolNotAllowed(ValueError):
    pass


class ToolDispatcher:
    """Only the three spec tools. No URL fetch, shell, or approve."""

    def __init__(self, retriever):
        self.retriever = retriever

    def call(self, name: str, **kwargs):
        if name not in ALLOWED_TOOLS:
            raise ToolNotAllowed(name)
        return getattr(self, name)(**kwargs)

    def search_chunks(self, query: str):
        return self.retriever.search(query)

    def get_chunk(self, chunk_id: str):
        return self.retriever.get_chunk(chunk_id)

    def get_document_meta(self, document_id: str):
        return self.retriever.get_document_meta(document_id)
