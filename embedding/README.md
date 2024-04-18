## Parallelization of Vertex AI embedding service

### Introduction

This class enables of parallelization of embedding calls when using the Vertex AI service (although the class could be easily expanded to work with other embedding services and models).

I wrote this class as part of a recent RAG project I worked on. The application relied heavily on the embedding service, as it allowed users to index new documents on the fly, and plus it used SemanticChunker (which requires the generation of embedding vectors for all sentences in the document prior to the chunking). So we had to come up with a way to improve the embedding process.

The class will dynamically batch all the documents to be embedded and asynchronously split the workload between the different services.  

### How to use it

Define your embedding services in the ```embedding_services.py``` file:

```python
VERTEX_EMBEDDING_MODEL_NAME = 'textembedding-gecko@003'

credentials = Credentials.from_service_account_file("<PATH_TO_SERVICE_ACCOUNT_JSON_1>")
EMBEDDING_SERVICE_1 = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING_MODEL_NAME, credentials=credentials,
                                         project="<NAME_OF_PROJECT_1>")
 ...                                        
```
Define as many services you have access to. These services can point to different GCP projects (where each project will have their respective service_account JSON), or use one single project (although you may end up hitting a rate limit if you define too many services pointing to the same project).

Then initialize the ParallelEmbeddings class by passing a list of all the embedding services you defined:

```python
EMBEDDING_SERVICES = [EMBEDDING_SERVICE_1, EMBEDDING_SERVICE_2, ...]
PARALLEL_EMBEDDING_SERVICE = ParallelVertexAIEmbeddings(model_name=VERTEX_EMBEDDING_MODEL_NAME,
                                                        embedding_services=EMBEDDING_SERVICES)
```

Finally, you can use it as part of your favorite LangChain pipeline, just like any other LangChain embedding class.

