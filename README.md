

# 🛕 Temple Chatbot Using RAG-LLM

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot for the **Travancore Temple**, designed to provide accurate, concise, and context-grounded answers using temple rules, regulations, and FAQs stored in document form.

The chatbot integrates **semantic document retrieval** with a **Large Language Model (LLM)** to minimize hallucinations and ensure responses are strictly based on authorized content.

---

## 📌 Project Objective

The objective of this project is to build a **domain-specific chatbot** that:
- Answers user queries using **temple documentation**
- Avoids hallucinations common in standalone LLMs
- Provides short, factual responses
- Responds with *“I don’t know”* for unrelated questions

---

## 🧠 Why RAG-LLM?

Traditional LLMs rely only on pre-trained knowledge and may generate incorrect or fabricated answers.
**RAG (Retrieval-Augmented Generation)** improves reliability by:

- Retrieving relevant document content before answering
- Grounding responses in external knowledge
- Eliminating the need for model fine-tuning
- Supporting easy updates by modifying documents

---

## 🏗️ System Architecture

The system consists of the following components:

1. **Document Loader**
   - Loads temple rules and FAQs from PDF files

2. **Text Splitter**
   - Breaks large documents into smaller semantic chunks

3. **Embedding Model**
   - Converts text chunks into vector embeddings using **Amazon Titan**

4. **Vector Store (FAISS)**
   - Stores embeddings and performs similarity search

5. **Retriever**
   - Fetches the most relevant document chunks for a query

6. **LLM (Claude via AWS Bedrock)**
   - Generates responses using retrieved context

7. **FastAPI Backend**
   - Exposes the chatbot as a REST API

---

## 🔄 Workflow

1. Temple documents are loaded and split into chunks
2. Chunks are converted into embeddings and stored in FAISS
3. User submits a question via API
4. Relevant document chunks are retrieved
5. Retrieved content is injected into the LLM prompt
6. LLM generates a concise, grounded response

---

## 🧩 Tech Stack

- **Programming Language:** Python
- **Framework:** FastAPI
- **LLM:** Claude 3 Haiku (AWS Bedrock)
- **Embeddings:** Amazon Titan
- **Vector Database:** FAISS
- **Libraries:** LangChain (Runnables), Boto3
- **API Testing:** Postman

---

## 📂 Project Structure

```

Temple-Chatbot-Using-RAG-LLM/

│
├── app.py # FastAPI application
├── rag_chain.py # RAG pipeline logic
├── loader.py # Document loading and vector store handling
├── requirements.txt # Project dependencies
├── .gitignore
│
└── data/
└── Sabarimala_RAG_Text_Heavy_9_Pages.pdf

```

---

## 🚀 API Usage

### Endpoint

```

POST /chat

```

### Request Body
```json
{
  "question": "What is the dress code at Sabarimala temple?"
}

```

### Response

```json
{
  "answer": "Male pilgrims must wear black, blue, or saffron clothing. Upper body garments are avoided."
}

```

---

## 🧪 Error Handling & Safety

- If the question is unrelated to temple content, the chatbot responds:
    
    ```
    I don't know
    
    ```
    
- Responses are strictly limited to retrieved context
- No external or assumed knowledge is used

---

## ⚠️ Limitations

- Accuracy depends on document quality
- Retrieval latency may slightly increase response time

---

## 🔮 Future Enhancements

- Session-based conversation memory
- Support for multiple temples
- User interface integration
- Advanced retrieval ranking
- Multilingual support


## ✅ Conclusion

This project demonstrates how **RAG-LLM architecture** can be used to build reliable, domain-specific chatbots by combining semantic search with controlled language generation. The system ensures factual accuracy, reduces hallucinations, and provides a scalable solution for document-based question answering.
