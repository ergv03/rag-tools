from langchain_google_vertexai import VertexAIEmbeddings
from google.oauth2.service_account import Credentials

VERTEX_EMBEDDING_MODEL_NAME = 'textembedding-gecko@003'

credentials = Credentials.from_service_account_file("<PATH_TO_SERVICE_ACCOUNT_JSON_1>")
EMBEDDING_SERVICE_1 = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING_MODEL_NAME, credentials=credentials,
                                         project="<NAME_OF_PROJECT_1>")

credentials = Credentials.from_service_account_file("<PATH_TO_SERVICE_ACCOUNT_JSON_2>")
EMBEDDING_SERVICE_2 = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING_MODEL_NAME, credentials=credentials,
                                         project="<NAME_OF_PROJECT_2>")
