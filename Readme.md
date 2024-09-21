## Gradio-based Document Chat Application Using LlamaIndex
`Note: I took the sample pdf from official LLamaIndex site so that we can ask question to LLM which is currently not known by LLM(ChatGPT).`

This document outlines the architecture and functionality of a Gradio-based application built with **LlamaIndex**, **OpenAI**, and **ChromaDB**. The application allows users to upload PDF or DOC files, which are then processed and indexed. Once indexed, the system can answer user queries based on the content of the uploaded documents, making it an example of a **Retrieval Augmented Generation (RAG)** system. 


### How to Install
First clone this repo using these commands - 
```
git clone https://github.com/HemantKArya/SimpRAGPipeline
cd SimpRAGPipeline

pip install virtualenv
python -m venv env
.\env\scripts\activate

pip install -r requirements.txt

python genai_rag.py
```


### Overview of the Pipeline

This system utilizes **LlamaIndex** to load, split, and process documents and embeds them using **OpenAI** models. The embeddings are stored in **ChromaDB**, enabling fast and efficient vector-based retrieval of relevant document sections when users ask questions. The application is wrapped in a **Gradio** interface, which provides a simple, interactive web-based UI.

### 1. **Packages and Libraries**

- **Gradio**: Provides the user interface for the system, enabling users to upload documents, input queries, and receive chat-based responses. Gradio simplifies the deployment of machine learning models by providing a browser-based interface.
  
- **LlamaIndex**: The core library that handles document reading, processing, splitting, embedding, and querying. It integrates directly with OpenAI models to generate responses based on retrieved document embeddings.

- **ChromaDB**: A vector database used to store and retrieve embeddings generated from document processing. It enables fast retrieval of relevant documents for answering queries.

- **OpenAI API**: Provides the language models used for embedding the document content and for generating conversational responses based on retrieved content.

- **tiktoken**: A tokenizer used for tokenizing input text for the OpenAI models.

- **os & shutil**: Used to handle file management and save uploaded documents.

### 2. **Pipeline Breakdown**

#### Step 1: **Directory Setup for Document Storage**

```python
doc_store_path = os.path.join(os.path.dirname(__file__), "doc_dir")
if not os.path.isdir(doc_store_path):
    os.makedirs(doc_store_path)
```

- **Document Directory**: The application checks if a `doc_dir` directory exists and creates it if necessary. This directory is where uploaded documents are stored locally.

#### Step 2: **Setting Up LLM and Memory Buffers**

```python
sum_llm = OpenAIsum(api_key=openai_api, model="gpt-3.5-turbo", max_tokens=256)
chat_summary_memory = ChatSummaryMemoryBuffer.from_defaults(
    token_limit=256,
    chat_store=chat_store,
    chat_store_key="user1",
    llm = sum_llm,
    tokenizer_fn = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
```

- **OpenAI LLM**: The **GPT-3.5-Turbo** model is used both for chat generation and summarization. This model is used for generating conversational responses to user queries.
  
- **Memory Buffer**: A chat memory buffer is used to store chat summaries, ensuring that the conversation’s context is retained across interactions.

#### Step 3: **ChromaDB and Vector Storage**

```python
db = chromadb.PersistentClient(path="./vec_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

- **ChromaDB**: ChromaDB acts as the persistent vector store where embeddings are saved. The embeddings for document chunks are stored in ChromaDB for fast similarity-based retrieval.

#### Step 4: **Vector Index Creation**

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
```

- **VectorStoreIndex**: This index connects the vector store (ChromaDB) to the query engine, enabling the retrieval of relevant document embeddings during queries.

#### Step 5: **Document Upload and Processing**

```python
def filter_unsaved(file_paths:list):
    for i in file_paths:
        if os.path.isfile(os.path.join(doc_store_path,os.path.basename(i))):
            file_paths.remove(i)
        else:
            shutil.copy2(i,doc_store_path)
    return file_paths
```

- **File Filtering**: This function checks if an uploaded file already exists in the local directory. If the file is new, it gets copied to the `doc_dir`.

```python
def add_doc(file_paths:list):
    file_paths = filter_unsaved(file_paths)
    if len(file_paths) == 0:
        return
    docs = SimpleDirectoryReader(input_files=file_paths).load_data()
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model,chunk_size=256)
    nodes = splitter.get_nodes_from_documents(docs)
    vector_index.insert_nodes(nodes)
```

- **Document Loading and Processing**: This function loads new documents, splits them into semantically meaningful chunks using **SemanticSplitterNodeParser**, and then generates embeddings for each chunk using OpenAI’s embedding model. These embeddings are then stored in the vector index for later retrieval.

#### Step 6: **Query Engine Setup**

```python
query_engine = vector_index.as_chat_engine(chat_memory=chat_summary_memory, storage_context=storage_context, use_async=True, similarity_top_k=2)
```

- **Query Engine**: The query engine is configured to handle incoming user queries by searching the vector index for relevant chunks of the uploaded documents. It uses the chat memory for retaining conversation context and retrieves up to 2 top results based on similarity.

#### Step 7: **Gradio Interface Setup**

```python
CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
```

- **Gradio UI Styling**: Custom CSS ensures that the interface is user-friendly and responsive, filling the screen space appropriately.

```python
def new_chat(chatbot:gr.Chatbot,textbox):
    query_engine.reset()
    return gr.update(value=""),[],"",gr.File(visible=False),gr.File(visible=False)
```

- **New Chat Function**: Resets the query engine when a new chat is initiated, clearing the conversation history and resetting the file display elements.

```python
def chat(history, input):
    response = query_engine.chat(str(input))
    files = []
    current_refs = ""
    for node in response.source_nodes:
        # Build reference details
        ...
        files.append({'path':node.metadata['file_path'],'show':True,})
    ...
    return gr.update(value=""),history + [(input, response.response)],current_refs,gr.update(visible=files[0]['show'],value=files[0]['path']),gr.update(visible=files[1]['show'],value=files[1]['path'])
```

- **Chat Function**: This function sends the user's input to the query engine, retrieves the response and source nodes (i.e., document sections). It compiles the references and displays up to two relevant files for download.

#### Step 8: **File Upload Handling**

```python
def file_upload(file,chatbot):
    add_doc(file)
    return gr.update(value="ChatDoc"),chatbot
```

- **File Upload**: When a user uploads a document, this function adds the document to the system, processes it, and updates the chat interface.

#### Step 9: **Gradio Interface**

```python
with gr.Blocks(fill_height=True, css=CSS) as demo:
    ...
    files.upload(file_upload,[files,chatbot],[title,chatbot])
```

- **Main Interface**: This block sets up the Gradio UI, where users can:
  - Upload files (PDFs or DOCs).
  - Input queries in the chatbox.
  - View document references and download related documents.
  
### 3. **Key Concepts**

- **RAG (Retrieval Augmented Generation)**: The system combines document retrieval (based on embeddings) with generative language models to answer user queries. It retrieves the most relevant sections of the uploaded documents before generating a response.

- **Document Embedding and Retrieval**: Documents are processed and split into chunks, with each chunk embedded using **OpenAI's embedding model**. These embeddings are stored in **ChromaDB** for fast retrieval during user queries.

- **Gradio Integration**: Gradio provides a simple web-based interface for interacting with the system. Users can upload documents, input queries, and download related document sections.

### 4. **End-to-End Workflow**

1. **Document Upload**: The user uploads PDF or DOC files to the system.
2. **Document Processing**: The documents are split into smaller semantic chunks and embedded using OpenAI models.
3. **Vector Store Indexing**: The embeddings are stored in **ChromaDB** for efficient retrieval.
4. **Query Processing**: When a user asks a question, the query engine retrieves the most relevant document chunks and generates a response based on the content.
5. **Chat Interaction**: Users interact with the chatbot interface, receiving responses to their queries and downloading relevant document sections if necessary.

### 5. **Conclusion**

This Gradio-based RAG system efficiently handles document upload, embedding, retrieval, and chat-based interaction. It leverages **OpenAI's GPT models** and **ChromaDB** for vector storage and retrieval, providing a flexible and powerful tool for answering document-based queries.