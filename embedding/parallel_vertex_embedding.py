from tqdm.asyncio import tqdm_asyncio
from typing import List
from random import choice
import asyncio
from langchain_google_vertexai import VertexAIEmbeddings
from typing import Any
import logging

logger = logging.getLogger(__name__)
MAX_RETRIES = 3
BATCH_SIZE = 250


async def async_embed_docs(texts, model):
    """
    Helper embedding docs function, compliant with asyncio
    """
    retries_so_far = 0
    max_retries = MAX_RETRIES
    # GCP Embedding service will sometimes return http 500
    while retries_so_far < max_retries:
        try:
            return await model.client.get_embeddings_async(texts)
        except Exception as e:
            logger.error(f'Error while async embedding docs. Error details: {e}')
            retries_so_far += 1

    raise logger.error(f'Error while async embedding docs: max number of retries reached ({max_retries}).')


class ParallelVertexAIEmbeddings(VertexAIEmbeddings):

    loop: Any = None
    embedding_services: list[VertexAIEmbeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        if not self.loop:
            self.loop = asyncio.new_event_loop()

        vectors = self.loop.run_until_complete(self._embed_documents(texts))
        to_return = []
        for vector in vectors:
            to_return += vector
        return [vector.values for vector in to_return]

    async def _embed_documents(self, texts: List[str]) -> List[List[float]]:

        tasks = []
        batch_size = BATCH_SIZE
        number_of_services = len(self.embedding_services)
        # Split the documents to be embedded between the different services
        docs_per_service = round(len(texts) / number_of_services) + 1

        for model_index in range(number_of_services):
            start_service = model_index * docs_per_service
            end_service = start_service + docs_per_service
            service_batches = self._prepare_batches(texts[start_service:end_service], batch_size)
            for batch in service_batches:
                task = asyncio.ensure_future(async_embed_docs(batch, self.embedding_services[model_index]))
                tasks.append(task)

        return await tqdm_asyncio.gather(*tasks)

    def embed_query(self, text: str) -> List[float]:

        model_to_use = choice(self.embedding_services)
        embeddings = model_to_use.client.get_embeddings([text])
        return embeddings[0].values
