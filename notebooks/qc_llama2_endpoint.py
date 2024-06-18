# Databricks notebook source
# MAGIC %pip install databricks-genai
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow.deployments

context = dbutils.entry_point.getDbutils().notebook().getContext()
token = context.apiToken().get()
host = context.apiUrl().get()

# Set the environment variables
os.environ['DATABRICKS_HOST'] = host
os.environ['DATABRICKS_TOKEN'] = token

client = mlflow.deployments.get_deploy_client("databricks")

completions_response = client.predict(
    endpoint="databricks-mpt-30b-instruct",
    inputs={
        "prompt": "What is the capital of France?",
        "temperature": 0.1,
        "max_tokens": 10,
        "n": 1  # Change the value of "n" to 1
    }
)

# COMMAND ----------

completions_response

# COMMAND ----------

chat_response = client.predict(
    endpoint="databricks-dbrx-instruct",
    inputs={
        "messages": [
            {
              "role": "user",
              "content": "Hello!"
            },
            {
              "role": "assistant",
              "content": "Hello! How can I assist you today?"
            },
            {
              "role": "user",
              "content": "What is a mixture of experts model??"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 20
    }
)

# COMMAND ----------

chat_response

# COMMAND ----------

chat_response = client.predict(
    endpoint="cjc-llama2",
    inputs={
        "messages": [
            {
              "role": "user",
              "content": "Hello!"
            },
            {
              "role": "assistant",
              "content": "Hello! How can I assist you today?"
            },
            {
              "role": "user",
              "content": "What is a mixture of experts model??"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 20
    }
)

# COMMAND ----------

chat_response

# COMMAND ----------


