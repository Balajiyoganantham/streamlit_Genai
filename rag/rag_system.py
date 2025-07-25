import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA # custom prompt templates for the language model
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter # sentence_splitter
from rag.PromptGenerator import PromptGenerator, PROMPTING_METHODS

load_dotenv() # load environment variables

class RAGSystem:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model="deepseek-r1-distill-llama-70b",
            temperature=0,  #Ensures deterministic output (no randomness).
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.document_path = "sample_document.txt"
        self.chunking_methods = {
            "fixed_size": RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            ),
            "sentence_splitter": CharacterTextSplitter(
                separator=". ",
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            ),
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
        self.vector_stores = {}
        self.qa_chains = {}
        self.load_and_process_document()

    def load_and_process_document(self):
        try:
            loader = TextLoader(self.document_path)
            documents = loader.load()
            for method_name, splitter in self.chunking_methods.items():
                chunks = splitter.split_documents(documents)
                vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.vector_stores[method_name] = vector_store
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff", # stuff: pass the entire context to the language model
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template="""Use the following pieces of context to answer the question at the end. \
                                        If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:""", # prompt template
                            input_variables=["context", "question"]
                        )
                    },
                    input_key="question"
                )
                self.qa_chains[method_name] = qa_chain
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            raise

    def get_chunking_analysis(self):
        analysis = {}
        try:
            loader = TextLoader(self.document_path)
            documents = loader.load()
            for method_name, splitter in self.chunking_methods.items():
                chunks = splitter.split_documents(documents)
                chunk_lengths = [len(chunk.page_content) for chunk in chunks]
                analysis[method_name] = {
                    "total_chunks": len(chunks),
                    "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                    "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                    "sample_chunk": chunks[0].page_content[:200] + "..." if chunks else ""
                }
        except Exception as e:
            logging.error(f"Error in chunking analysis: {str(e)}")
            analysis = {"error": str(e)}
        return analysis

    def get_prompting_methods(self):
        return PROMPTING_METHODS

    def query_with_method(self, question, method_name, prompt_method=None, custom_prompt=None):
        try:
            if method_name not in self.vector_stores:
                return {"error": f"Method {method_name} not found"}
            # Load the document context
            loader = TextLoader(self.document_path)
            documents = loader.load()
            context = documents[0].page_content if documents else ""

            # Choose prompt template
            from rag.PromptGenerator import PromptGenerator
            if not prompt_method or prompt_method == 'default':
                prompt_template = PromptGenerator.create_default_prompt(context)
            else:
                prompt_template = PromptGenerator.create_prompt_by_method(context, prompt_method)

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Build a new QA chain with the selected prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_stores[method_name].as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
                input_key="question"
            )

            result = qa_chain({"question": question})
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "method": method_name
            }
        except Exception as e:
            logging.error(f"Error querying with method {method_name}: {str(e)}")
            return {"error": str(e)} 