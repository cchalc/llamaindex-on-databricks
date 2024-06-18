# Databricks notebook source
# MAGIC %md # Debug

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()  # noqa

# COMMAND ----------

import os
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from pathlib import Path
from llama_index import download_loader
from llama_index import set_global_service_context

from mlflow.deployments import get_deploy_client

# from rag_demo.basic_rag import DatabricksLLM, DatabricksEmbedding
# from rag_demo.vectorstore import DatabricksVectorStore

from databricks_llamaindex.databricks_llm import DatabricksLLM, DatabricksEmbedding
from databricks_llamaindex.databricks_vector_search import DatabricksVectorStore

# COMMAND ----------

context = dbutils.entry_point.getDbutils().notebook().getContext()
token = context.apiToken().get()
host = context.apiUrl().get()

# Set the environment variables
os.environ['DATABRICKS_HOST'] = host
os.environ['DATABRICKS_TOKEN'] = token

# COMMAND ----------

databricks_embedding_model = DatabricksEmbedding(endpoint="databricks_e5_v2")

service_context = ServiceContext.from_defaults(
    llm=DatabricksLLM(endpoint="cjc_llama2"),
    embed_model=databricks_embedding_model,
    context_window=512,
    num_output=256,
)
set_global_service_context(service_context)

# COMMAND ----------

dvs = DatabricksVectorStore(
    endpoint="one-env-shared-endpoint-8",
    index_name="cjc.scratch.bbc_news_train_index",
    host=host,
    token=token,
    text_field="Text",
    embedding_field="embedding",
    id_field="ArticleId",
)

# COMMAND ----------

dvs_index = VectorStoreIndex.from_vector_store(dvs)
query_engine = dvs_index.as_query_engine()

# Query and print response
response = query_engine.query("Are there any weak points in ChatGPT for Zero Shot Learning?")
print(response)

# COMMAND ----------

def main_simple_rag():
    # client = get_deploy_client("databricks")
    # print(client.list_endpoints())
    databricks_embedding_model = DatabricksEmbedding(endpoint="databricks_e5_v2")
    service_context = ServiceContext.from_defaults(
        llm=DatabricksLLM(endpoint="cjc_llama2"),
        embed_model=databricks_embedding_model,
        context_window=512,
        num_output=256,
    )
    set_global_service_context(service_context)
    PyMuPDFReader = download_loader("PyMuPDFReader")

    loader = PyMuPDFReader()

    # https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#notebook/732197971898460/command/732197971898681
    # ANZ_LLM_Bootcamp
    input_folder = "/Volumes/cjc/datasets/raw_data" 
    documents = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            documents.extend(
                loader.load_data(
                    file_path=Path(os.path.join(root, file)), metadata=True
                )
            )
    s_index = SummaryIndex.from_documents(documents, service_context=service_context)
    docs_list = list(s_index.docstore.docs.values())
    for doc in docs_list:
        doc.embedding = databricks_embedding_model
    # dvs = DatabricksVectorStore(
    #     "shared-demo-endpoint",
    #     "msh.test.test_direct_vector_index",
    #     "https://e2-demo-field-eng.cloud.databricks.com",
    #     os.environ["DATABRICKS_TOKEN"],
    #     "field2",
    #     "text_vector",
    # )
    dvs = DatabricksVectorStore(
        endpoint="dbdemos_vs_endpoint",
        index_name="cjc.scratch.rag_debug",
        host="https://e2-demo-field-eng.cloud.databricks.com",
        token=os.environ["DATABRICKS_TOKEN"],
        text_field="content",
        embedding_field="embedding",
        id_field="id",
    )
    # dvs.add(docs_list)
    dvs_index = VectorStoreIndex.from_vector_store(dvs)
    query_engine = dvs_index.as_query_engine()

    # Query and print response
    response = query_engine.query("Are there any weak points in ChatGPT for Zero Shot Learning?")
    print(response)

# COMMAND ----------

main_simple_rag()

# COMMAND ----------


