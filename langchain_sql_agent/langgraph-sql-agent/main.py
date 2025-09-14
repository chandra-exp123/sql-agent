## main.py
# pip install -U langgraph langchain langchain-community langchain-google-genai sqlalchemy oracledb

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

# --- Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=1024
)

# --- Connect to Database ---
# SQLite connection (local file)
db_uri = "sqlite:///Chinook.db"
db = SQLDatabase.from_uri(db_uri)
print("Dialect:", db.dialect)
print("Tables:", db.get_usable_table_names())

# --- Setup toolkit (tools to interact with DB) ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# --- System prompt guiding the agent ---
system_prompt = f"""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.
You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.
You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.
Do NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.
Always start by looking at the tables in the database to see what you can query,
then query the schema of the most relevant tables.
"""

# --- Create the agent ---
agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)

# --- Example query ---
question = "Which employees were hired most recently?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
