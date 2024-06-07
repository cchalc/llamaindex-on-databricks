# Databricks notebook source
# MAGIC %md #Using OSS Models served from Marketplace and Databricks Vector Search with LlamaIndex

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()  # noqa

# COMMAND ----------

from llama_index import ServiceContext, VectorStoreIndex, set_global_service_context

from databricks_llamaindex.databricks_llm import DatabricksLLM, DatabricksEmbedding
from databricks_llamaindex.databricks_vector_search import DatabricksVectorStore

# COMMAND ----------

context = dbutils.entry_point.getDbutils().notebook().getContext()  # noqa
token = context.apiToken().get()
host = context.apiUrl().get()

service_context = ServiceContext.from_defaults(
    llm=DatabricksLLM(endpoint="cjc-llama2"),
    embed_model=DatabricksEmbedding(endpoint="databricks_e5_v2"),
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
# dvs.add(docs_list)
dvs_index = VectorStoreIndex.from_vector_store(dvs)
query_engine = dvs_index.as_query_engine()


# COMMAND ----------

# Query and print response
response = query_engine.query("What is Unity Catalog?")
print(response)

# COMMAND ----------


