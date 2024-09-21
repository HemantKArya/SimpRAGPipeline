## Retrieval Augmented Generation (RAG) based Nodebook Using LlamaIndex

`Note: I took the sample pdf from official LLamaIndex site so that we can ask question to LLM which is currently not known by LLM(ChatGPT).`

This document explains the design and implementation of a Retrieval Augmented Generation (RAG) based program in Python, utilizing the **LlamaIndex** library. This notebook retrieves and answers questions based on content from an uploaded PDF file. The pipeline uses **OpenAI models** to process and embed documents and queries, and **ChromaDB** as the persistent storage for embeddings. This guide provides an in-depth explanation of the entire workflow, including the packages used, and the key functions involved.


### How to Use
First clone this repo using these commands - 
```
git clone https://github.com/HemantKArya/SimpRAGPipeline
cd SimpRAGPipeline

```
```
#remember to replace your openai api key inside notebook here at top - 

openai_api = "OPEN_API_KEY"    <----

```
and then run this in cmd -
```
pip install virtualenv
python -m venv env
.\env\scripts\activate

pip install -r requirements.txt
pip install notebook
jupyter notebook

```
And then navigate to current folder and open this 'genai_rag.ipynb'.

### Overview of the Pipeline

This program reads documents (PDFs in this case), processes them to create embeddings, stores these embeddings in a vector database, and then uses these embeddings to provide accurate and relevant responses to user queries. This system is built using **LlamaIndex**, **OpenAI**, **ChromaDB**.

### 1. **Packages and Libraries**

- **Streamlit**: Used for building user interfaces. In this case, it stores the OpenAI API key in a secrets management system and would provide the user interface if expanded.
  
- **LlamaIndex**: The core library for loading, processing, and querying documents. It manages:
  - **Node Parsers**: These split and process the documents into manageable chunks.
  - **Vector Store Indexes**: These store the embeddings of processed documents.
  - **LLM Integration**: LlamaIndex integrates with OpenAI LLMs to handle queries.

- **ChromaDB**: A vector database used to persist embeddings and vectorized data. It serves as the persistent storage for document embeddings.

- **OpenAI API**: Used for text generation and embedding creation through models such as GPT-3.5-Turbo and Ada-002. This API is responsible for:
  - Generating responses based on user queries.
  - Embedding documents to create vector representations.

- **tiktoken**: A tokenizer library that converts text into tokens for the OpenAI models.

- **os & shutil**: Used for file system operations, ensuring that documents are loaded and saved appropriately.

### 2. **Pipeline Breakdown**

The pipeline comprises several stages, beginning with the reading and processing of documents, followed by the indexing and querying steps.

#### Step 1: **Setting Up Chat Memory and Chat Store**

```python
chat_store = SimpleChatStore()
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1"
)
```

- **Chat Store and Chat Memory**: These objects are used to track and manage the conversational history of the user. **SimpleChatStore** stores previous interactions, while **ChatMemoryBuffer** retains recent interactions in memory, ensuring the conversation context is available.

#### Step 2: **Summarizing Chat Using LLM (OpenAI)**

```python
sum_llm = OpenAIsum(api_key=openai_api, model="gpt-3.5-turbo", max_tokens=256)
chat_summary_memory = ChatSummaryMemoryBuffer.from_defaults(
    token_limit=256,
    chat_store=chat_store,
    chat_store_key="user1",
    llm=sum_llm,
    tokenizer_fn=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
```

- **OpenAI LLM**: The **GPT-3.5-Turbo** model is used for chat summarization. This allows the conversation history to be summarized into tokens for memory management. **tiktoken** is used to efficiently tokenize inputs and manage the maximum token count.

#### Step 3: **Loading and Persisting Vector Data**

```python
db = chromadb.PersistentClient(path="./vec_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

- **ChromaDB**: ChromaDB acts as the persistent vector store. The document embeddings generated in the later steps are stored in this vector database, enabling efficient retrieval during querying.

#### Step 4: **Creating a Vector Store Index**

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)
```

- **VectorStoreIndex**: The **VectorStoreIndex** creates and manages a connection between the document embeddings and the vector store (ChromaDB in this case). This index helps facilitate fast similarity searches when responding to user queries.

#### Step 5: **Document Loading and Processing**

```python
documents = SimpleDirectoryReader("./data").load_data()
```

- **SimpleDirectoryReader**: This utility reads documents (PDFs) from a specified directory (`./data`). It loads these documents into memory for further processing.

#### Step 6: **Adding and Processing New Documents**

```python
def add_doc(file_paths:list):
    file_paths = filter_unsaved(file_paths)
    if len(file_paths) == 0:
        return
    docs = SimpleDirectoryReader(input_files=file_paths).load_data()
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model, chunk_size=256)
    nodes = splitter.get_nodes_from_documents(docs)
    vector_index2 = VectorStoreIndex(nodes)
    vector_index.insert_nodes(nodes)
```

- **Adding Documents**: New documents are processed and stored through this function. The **filter_unsaved** function checks if the documents already exist in the local directory before proceeding to add new ones.
  
- **Document Splitting**: The **SemanticSplitterNodeParser** breaks down documents into smaller semantic chunks that can be processed individually. This chunking ensures that each part of the document is adequately indexed for retrieval.
  
- **Vector Embeddings**: Once the documents are processed, embeddings are created using the **OpenAIEmbedding** model (`text-embedding-ada-002`). These embeddings are then inserted into the vector store (ChromaDB).

#### Step 7: **Querying the Vector Store**

```python
response = q_engine.chat("what is llamaindex?")
print(response.response)

response = q_engine.chat("tell me how to make simple llama project in python?")
print(response.response)
```

- **Querying the System**: Queries can be made through the **vector_query_engine** object. This engine looks for relevant embeddings in the vector store based on the user's input and retrieves the corresponding document sections.

#### Step 8: **Filtering Duplicate Files**

```python
def filter_unsaved(file_paths:list):
    for i in file_paths:
        if os.path.isfile(os.path.join(doc_store_path, os.path.basename(i))):
            file_paths.remove(i)
            print("File already exist: {}".format(i))
        else:
            shutil.copy2(i, doc_store_path)
    return file_paths
```

- **File Checking**: This function ensures that no duplicate files are added to the document directory. If a file already exists, it is not reprocessed, improving efficiency.

### 3. **Key Concepts**

- **RAG (Retrieval Augmented Generation)**: The system retrieves relevant information from the document embeddings before generating a response. This retrieval step ensures the model has access to accurate data, making the output more reliable.
  
- **Embeddings and Vector Search**: Embeddings are vector representations of text, enabling the system to perform similarity searches. By storing these embeddings in ChromaDB, the system can quickly retrieve relevant sections of documents based on the user’s query.
  
- **Node Parsing and Chunking**: Large documents are broken down into smaller, meaningful chunks or "nodes" to make embedding and retrieval more manageable. This chunking is done semantically, so each piece of text is coherent and useful in isolation.

### 4. **End-to-End Workflow**

1. **Document Upload**: PDFs are uploaded to the system and stored in the local directory (`./data`).
2. **Embedding Creation**: The document is split into smaller chunks, and embeddings are created using OpenAI's embedding model.
3. **Vector Store**: Embeddings are stored in ChromaDB, a persistent vector database.
4. **Query Processing**: When a user asks a question, the system searches the embeddings for relevant information and generates a response using OpenAI's GPT model.
5. **Chat Memory**: The system maintains a conversation history to keep the context, summarizing it as necessary to manage token limits.

### 5. **Conclusion**

This RAG-based Python notebook is a robust tool for answering questions based on document contents, with a clear flow from document ingestion to query response generation. It uses state-of-the-art technologies like LlamaIndex and OpenAI models, underpinned by ChromaDB for efficient storage and retrieval of document embeddings.