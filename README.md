# AmbedkarGPT – Intern Task (RAG Pipeline)

A fully local Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, HuggingFace embeddings, and a local LLM served via LM Studio.  
No API keys, no cloud — 100% offline.

---

## Features
- Local RAG pipeline  
- ChromaDB vector store  
- MiniLM embeddings  
- LM Studio as LLM server  
- Clean CLI interface  
- Context-grounded answers   

---

## Project Structure

```bash
AmbedkarGPT-Intern-Task/
│
├── main.py
├── requirements.txt
├── README.md
└── speech.txt
```

---

## Setup
### 1. Clone
```bash
git clone https://github.com/your-username/AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
```
### 2. Create & Activate Conda Env
```bash
conda create -n ambedkargpt python=3.10 -y
conda activate ambedkargpt
```

### 3. Install Dependencies
 ```bash
pip install -r requirements.txt
```
---

## LM Studio Setup
- Load a model (e.g., mistralai/mistral-7b-instruct-v0.3)
- Start local server: Developer → Start Local Inference Server
- Note the API URL (e.g., http://localhost:1234/v1)
- Update main.py if needed
---
## Run

```bash
python main.py
```

---

## Output

<img width="1467" height="472" alt="Screenshot 2025-11-18 130601" src="https://github.com/user-attachments/assets/dfb6b476-d442-46d7-a9cf-2c9d19dce97d" />

---
## Test Questions

### Basic
```
 - What does Ambedkar say is the real remedy?
 - Why does he ask people to take a stand against the scriptures?
 - What two options does society need to choose between?
```
### Intermediate
```
 - Why does he compare social reform to pruning leaves?
 - Why do social reformers fail to eliminate caste?
```
### Control (should return “I don’t know”)
```
 - What does Ambedkar say about democracy?
 - What year was the speech written?
 - What does he say about Buddhism?
```
---
## Data Source

The file `speech.txt` includes the text used for retrieval.

