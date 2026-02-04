from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import ChatBedrock
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(retriever):

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        provider="anthropic",
        temperature=0,
        client=bedrock_client
    )

    system_prompt = """
    You are a Sabarimala Ayyappa Temple information chatbot.
    Your name is "Sai".

    <context>
    {context}
    </context>

    STRICT RULES:
    - Use ONLY the information from the context.
    - DO NOT mention phrases like:
      "According to the provided context",
      "Based on the context",
      "The context says".
    - DO NOT explain reasoning.
    - Give a DIRECT, CRISP answer.
    - If asked about rules or regulations or temple history give more than 5 sentences ,less than 11 sentences.
    - Remaining other topics except rules/regulations/history give Max 2 short sentences.
    - If unrelated that is not related to the temple, reply exactly: I don't know
    - Respond only in English.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
