import gradio as gr
import time
import os
import shutil
openai_api = "OPEN_API_KEY"

doc_store_path = os.path.join(os.path.dirname(__file__), "doc_dir")
if not os.path.isdir(doc_store_path):
    os.makedirs(doc_store_path)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai import OpenAI as OpenAIsum
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatSummaryMemoryBuffer

import chromadb
import tiktoken


chat_store = SimpleChatStore()
# chat_memory = ChatMemoryBuffer.from_defaults(
#     token_limit=3000,
#     chat_store=chat_store,
#     chat_store_key="user1",
# )


sum_llm = OpenAIsum(api_key=openai_api, model="gpt-3.5-turbo", max_tokens=256)
chat_summary_memory = ChatSummaryMemoryBuffer.from_defaults(
    token_limit=256,
    chat_store=chat_store,
    chat_store_key="user1",
    llm = sum_llm,
    tokenizer_fn = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)


chat_store = SimpleChatStore.from_persist_path(
    persist_path="chat_store.json"
)



# documents = SimpleDirectoryReader("./data").load_data()
db = chromadb.PersistentClient(path="./vec_db")

chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.llm = OpenAI(model="gpt-3.5-turbo",api_key=openai_api,)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

vector_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context,)
query_engine = vector_index.as_chat_engine(chat_memory=chat_summary_memory,storage_context=storage_context,use_async=True,similarity_top_k=2)

current_refs = ""



def filter_unsaved(file_paths:list):
    for i in file_paths:
        if os.path.isfile(os.path.join(doc_store_path,os.path.basename(i))):
            file_paths.remove(i)
            print("File already exist : {}".format(i))
        else:
            shutil.copy2(i,doc_store_path)
    return file_paths

def add_doc(file_paths:list):
    print(file_paths)
    file_paths = filter_unsaved(file_paths)
    print(file_paths)
    if len(file_paths) == 0:
        return
    docs = SimpleDirectoryReader(input_files=file_paths).load_data()
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model,chunk_size=256)
    nodes = splitter.get_nodes_from_documents(docs)
    vector_index2 = VectorStoreIndex(nodes)
    vector_index.insert_nodes(nodes)
    




CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""


def new_chat(chatbot:gr.Chatbot,textbox):
    query_engine.reset()
    return gr.update(value=""),[],"",gr.File(visible=False),gr.File(visible=False)


def chat(history, input):
    response = query_engine.chat(str(input))
    global current_refs
    files = []
    current_refs = ""
    for node in response.source_nodes:
        try:
            current_refs += f"{str(node.metadata['title'])},"
        except:
            current_refs += ""
        try:
            current_refs += f"Pg - {str(node.metadata['page_label'])},"
        except:
            current_refs += "Pg - ,"
        try:
            current_refs += f"File - {str(node.metadata['file_name'])} \n,"
        except:
            current_refs += "File - ,\n"
        
        try:
            files.append({'path':node.metadata['file_path'],'show':True,})
        except:
            files.append({'path':None,'show':False,})

    if len(files) < 2:
        for _ in range(2-len(files)):
            files.append({'path':None,'show':False,})
        
    return gr.update(value=""),history + [(input, response.response)],current_refs,gr.update(visible=files[0]['show'],value=files[0]['path']),gr.update(visible=files[1]['show'],value=files[1]['path'])

def file_upload(file,chatbot):
    print(file)
    add_doc(file)
    return gr.update(value="ChatDoc"),chatbot

with gr.Blocks(fill_height=True, css=CSS) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            title = gr.Label(value="chatdoc", label="ChatDoc")
            files = gr.UploadButton(
                    "📁 Upload PDF or doc files", file_types=[
                        '.pdf',
                        '.doc'
                    ],
                    file_count="multiple")
            references = gr.Textbox(label="References",interactive=False)
            file_down1 = gr.File(visible=False)
            file_down2 = gr.File(visible=False) 
            

        
        with gr.Column(scale=9,):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                label="ChatDoc",
                avatar_images=["https://www.freeiconspng.com/thumbs/person-icon-blue/person-icon-blue-25.png","https://cdn-icons-png.flaticon.com/512/8943/8943377.png"],
            )
            with gr.Row():
                textbox = gr.Textbox(label="Type your message", scale=10)
                clear = gr.Button(value="New Chat", size="sm", scale=1)
                clear.click(new_chat,[],[textbox, chatbot,references,file_down1,file_down2])
                textbox.submit(chat, [chatbot, textbox], [textbox, chatbot,references,file_down1,file_down2])
                

        files.upload(file_upload,[files,chatbot],[title,chatbot])
                
    
demo.launch(share=True)