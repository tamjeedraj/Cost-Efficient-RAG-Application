from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def embed(self, text):

        return self.model.encode(text).tolist()

    def dimension(self):

        return self.model.get_sentence_embedding_dimension()