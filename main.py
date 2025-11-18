import os
import textwrap
from typing import List

# community imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")
# LM Studio / OpenAI-compatible client
from langchain_openai import ChatOpenAI

# ---------------------
# CONFIG
# ---------------------
SPEECH_PATH = "speech.txt"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set this to the base URL LM Studio shows (use your IP or localhost)
LM_STUDIO_API = "http://192.168.1.7:1234/v1"   # <- update if different
LM_STUDIO_MODEL = "mistralai/mistral-7b-instruct-v0.3"  # exact model name from LM Studio
LM_STUDIO_KEY = "lm-studio"  # LM Studio accepts any string here

# how many context chunks to retrieve
TOP_K = 3

# ---------------------
# helpers
# ---------------------
def build_or_load_chroma(persist_dir: str = CHROMA_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # build DB if missing
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        if not os.path.exists(SPEECH_PATH):
            raise FileNotFoundError(f"{SPEECH_PATH} not found. Put the speech text file in the project folder.")
        print("Building vector DB (this may take a moment, model download + embedding) ...")
        loader = TextLoader(SPEECH_PATH, encoding="utf-8")
        docs = loader.load()

        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()
        print("Vector DB built and persisted.")
        return vectordb

    # otherwise load existing
    print("Loading existing Chroma vector DB...")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb


def retrieve_top_chunks(vectordb: Chroma, query: str, k: int = TOP_K) -> List[str]:
    """
    Use Chroma's similarity_search (stable API) to get top-k documents (page_content strings).
    """
    # many Chroma wrappers provide .similarity_search; if not, .similarity_search_with_score exists
    try:
        docs = vectordb.similarity_search(query, k=k)
    except Exception:
        # fallback: use retriever if available
        try:
            retriever = vectordb.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            raise RuntimeError("Failed to run similarity search on Chroma: " + str(e))

    return [d.page_content for d in docs]


def make_prompt(context_chunks: List[str], question: str) -> str:
    """
    Construct a concise prompt that instructs the LLM to use ONLY the provided context.
    """
    context = "\n\n---\n\n".join(chunk.strip() for chunk in context_chunks if chunk and chunk.strip())
    prompt = textwrap.dedent(f"""
    You are given the following context snippets from a speech. Use ONLY the information in the context to answer the question.
    If the answer is not present in the context, say "I don't know based on the provided text."

    Context:
    {context}

    Question:
    {question}

    Answer fully and do not stop mid-sentence. 
    Finish your response completely before ending. 
    Cite (briefly) which snippet(s) you used, e.g. [snippet 1].
    Provide a complete, finished answer. 
    Do NOT end abruptly. 
    Ensure your final sentence is fully written.
    """).strip()
    return prompt


def get_llm():
    # ChatOpenAI works with LM Studio's OpenAI-compatible server
    llm = ChatOpenAI(
        model=LM_STUDIO_MODEL,
        api_key=LM_STUDIO_KEY,
        base_url=LM_STUDIO_API,
        temperature=0.1,
        max_tokens=512
    )
    return llm


def answer_query(vectordb: Chroma, llm: ChatOpenAI, question: str) -> dict:
    chunks = retrieve_top_chunks(vectordb, question, k=TOP_K)
    prompt = make_prompt(chunks, question)

    # Use LM Studio via OpenAI-compatible local API
    from openai import OpenAI
    client = OpenAI(base_url=LM_STUDIO_API, api_key="lm-studio")

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048

        )

        answer = response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Failed to call LLM: {e}")

    return {"answer": answer, "chunks": chunks}



# ---------------------
# CLI main
# ---------------------
def main():
    print("\n=== AmbedkarGPT (LM Studio) ===\n")
    vectordb = build_or_load_chroma()
    llm = get_llm()

    print("\nReady. Ask questions about the speech. Type 'exit' to quit.\n")
    while True:
        q = input("User Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print("\nRetrieving context and asking the model...\n")
        try:
            res = answer_query(vectordb, llm, q)
        except Exception as e:
            print("Error during query:", e)
            continue

        print("\nAnswer:\n")
        print(res["answer"].strip())
        print("\n")


if __name__ == "__main__":
    main()