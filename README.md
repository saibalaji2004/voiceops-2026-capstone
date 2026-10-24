\# VoiceOps 2026 Capstone

\### Agentic AI-Based Voice Customer Support System



VoiceOps is an AI-powered customer support assistant developed as a capstone project using a synthetic ZentroMart customer support dataset. The application combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), conversation memory, tool-based order verification, support ticket creation, and speech processing.



The project demonstrates the development of a customer support agent through four stages: a prompt-only baseline, a RAG-based assistant, a stateful agent with tools, and a voice-enabled interface.



\## Table of Contents



\- \[Project Overview](#project-overview)

\- \[Key Features](#key-features)

\- \[Four Implementation Stages](#four-implementation-stages)

\- \[Technology Stack](#technology-stack)

\- \[System Architecture](#system-architecture)

\- \[Project Structure](#project-structure)

\- \[Prerequisites](#prerequisites)

\- \[Installation and Configuration](#installation-and-configuration)

\- \[Running the Application](#running-the-application)

\- \[Validation and Evaluation](#validation-and-evaluation)

\- \[Evaluation Artifacts](#evaluation-artifacts)

\- \[Security and Data Handling](#security-and-data-handling)

\- \[Limitations](#limitations)



\## Project Overview



Traditional chatbots often struggle to maintain context, retrieve the correct policy information, verify customer identity, and perform customer support actions safely.



VoiceOps explores how an AI customer support system can address these challenges by combining several components:



\- LLM-based response generation.

\- Retrieval of relevant knowledge-base information.

\- Conversation history and follow-up handling.

\- Controlled access to mock order lookup and support ticket tools.

\- Caller verification before disclosing protected order information.

\- Speech-to-text (STT) and text-to-speech (TTS) integration.



The project uses a supplied synthetic dataset containing customer records, orders, products, policies, knowledge-base documents, practice questions, and conversation examples.



\## Key Features



\### 1. Prompt-Based Customer Support



A baseline chatbot generates responses using instructions and an LLM without the main RAG retrieval workflow.



\### 2. Retrieval-Augmented Generation



The RAG implementation retrieves relevant knowledge-base sections and product information to provide evidence-grounded customer support responses.



The retrieval evaluation includes comparisons across different values of `top\_k`.



\### 3. Stateful Agentic Customer Support



The agent implementation supports:



\- Conversation history and follow-up questions.

\- Order ID collection and order lookup.

\- Customer verification using the supported verification methods.

\- Restrictions on order disclosures before successful verification.

\- Limited verification retries.

\- Support ticket creation through the supplied mock tool.

\- Controlled handover messaging based on tool outcomes.



Order operations use the local synthetic dataset and mock tools, not a live commercial order-management system.



\### 4. Voice Interface



The voice module integrates Gnani speech services to:



1\. Transcribe an input audio file into text.

2\. Send the transcribed message to the customer support agent.

3\. Generate a spoken response using text-to-speech.

4\. Save the generated audio response to a file.



The English (India) voice workflow was tested successfully using an audio file.



\## Four Implementation Stages



| Stage | Implementation | Purpose |

|---|---|---|

| Step 1 | `app/chatbot.py` | Establish a prompt-only baseline |

| Step 2 | `app/rag.py`, `app/rag\_chatbot.py` | Retrieve relevant evidence for customer responses |

| Step 3 | `app/step3\_agent.py` | Maintain conversation state and use support tools safely |

| Step 4 | `app/voice.py` | Integrate speech-to-text and text-to-speech |



Supporting modules include the LLM provider router, evaluation scripts, retrieval audits, and the interactive application entry point.



\## Technology Stack



| Component | Technologies |

|---|---|

| Programming language | Python 3.11 |

| LLM integration | Configured Gemini, Groq, OpenRouter, and Cloudflare providers |

| Retrieval | Sentence Transformers, NumPy, semantic retrieval and ranking |

| Knowledge base | Markdown documents and product catalogue data |

| Agent tools | Python functions and CSV-based synthetic data |

| Voice processing | Gnani STT and TTS |

| Configuration | Environment variables and `.env` |

| Version control | Git and GitHub |



The availability of each LLM provider depends on its credentials, model configuration, provider limits, and service availability.



\## System Architecture



```text

&#x20;                Customer Message

&#x20;                       |

&#x20;                       v

&#x20;             Interactive Application

&#x20;                       |

&#x20;                       v

&#x20;               Stateful Agent

&#x20;                       |

&#x20;          +------------+-------------+

&#x20;          |                          |

&#x20;          v                          v

&#x20;     Conversation                 RAG Retrieval

&#x20;      History                    and Knowledge Base

&#x20;          |                          |

&#x20;          +------------+-------------+

&#x20;                       |

&#x20;                       v

&#x20;              LLM Response Router

&#x20;                       |

&#x20;                       v

&#x20;              Customer Response

&#x20;                       |

&#x20;           +-----------+-----------+

&#x20;           |                       |

&#x20;           v                       v

&#x20;      Order Lookup             Ticket Creation

&#x20;      and Verification         and Handover

&#x20;           |

&#x20;           v

&#x20;      Synthetic Dataset



Audio Input --> Gnani STT --> Agent --> Gnani TTS --> Audio Output

```



\## Project Structure



```text

voiceops-2026-capstone/

|

|-- app/

|   |-- chatbot.py

|   |-- config.py

|   |-- llm\_router.py

|   |-- main.py

|   |-- rag.py

|   |-- rag\_chatbot.py

|   |-- step3\_agent.py

|   |-- voice.py

|   |-- evaluate\_step1.py

|   |-- evaluate\_step2.py

|   |-- evaluate\_retrieval\_k.py

|   |-- compare\_evaluations.py

|   `-- audit\_step2\_retrieval.py

|

|-- Capstone\_Project\_Dataset\_FTTT\_2026/

|   |-- data/

|   |-- kb/

|   |-- practice/

|   |-- tools/

|   |-- README.md

|   |-- DATA\_DICTIONARY.md

|   `-- validate\_dataset.py

|

|-- outputs/

|   |-- rag\_index/

|   |-- retrieval\_k\_sweep.csv

|   |-- step1\_baseline.csv

|   |-- step1\_step2\_comparison.csv

|   |-- step2\_rag.csv

|   `-- voice\_test.wav

|

|-- .env.example

|-- .gitignore

|-- requirements.txt

`-- README.md

```



The `outputs/` directory contains generated retrieval indexes, evaluation artifacts, and the sample voice input. Additional output files may be generated during execution.



\## Prerequisites



\- Windows, macOS, or Linux with an appropriate Python environment.

\- Python 3.11 recommended for the tested environment.

\- Git for source control.

\- API credentials for the LLM provider or providers you intend to use.

\- Gnani API credentials for the voice workflow.

\- Internet access for the configured cloud-based model and speech services.



\## Installation and Configuration



\### 1. Clone the repository



```bash

git clone https://github.com/saibalaji2004/voiceops-2026-capstone.git

cd voiceops-2026-capstone

```



\### 2. Create a virtual environment



On Windows PowerShell:



```powershell

py -3.11 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script activation, use an appropriate environment activation method for your machine rather than changing system-wide security settings unnecessarily.



\### 3. Install dependencies



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 4. Configure environment variables



Create a local `.env` file from the provided template:



```powershell

Copy-Item .env.example .env

```



Run this copy command only when `.env` does not already exist. If you have already configured your credentials, do not overwrite your existing file.



Open `.env` and configure the credentials required by your selected providers. The template contains placeholder values for the available LLM providers and Gnani.



Keep actual credentials in `.env`, not in `README.md`, source code, screenshots, or `.env.example`.



\## Running the Application



\### Start the interactive customer support agent



From the project root:



```powershell

python -m app.main

```



Enter customer messages in the terminal and follow the prompts. The exact response depends on the supplied question, retrieved evidence, configured providers, and agent state.



\### Run the voice interface



Use the supplied audio test file:



```powershell

python -m app.voice "outputs\\voice\_test.wav" --language en-IN

```



The command transcribes the audio, processes the message through the agent, and saves the generated speech response to:



```text

outputs/voice\_reply.wav

```



The voice workflow requires valid Gnani credentials and a compatible audio input.



\### Validate the supplied dataset



```powershell

python Capstone\_Project\_Dataset\_FTTT\_2026\\validate\_dataset.py

```



\### Compile-check the Python modules



```powershell

python -m compileall -q app Capstone\_Project\_Dataset\_FTTT\_2026\\tools

```



\## Validation and Evaluation



The following checks were completed in the development environment:



\- Python compilation completed successfully.

\- The supplied dataset validator passed.

\- The dataset validation covered all 9 CSV files, 21 knowledge-base documents, primary and foreign keys, order calculations, practice evidence, verification gates, and company naming.

\- The validated dataset contains 1,000 orders, 1,783 order items, 244 support logs, 52 practice questions, and 23 conversations.

\- The order lookup and caller verification workflow was exercised successfully.

\- Support ticket creation was exercised using the mock tool.

\- The Gnani STT-to-agent-to-TTS workflow completed successfully for a sample English (India) audio input.



These results describe the checks actually performed in the development environment; they are not a claim that every possible conversation or production scenario has been tested.



\### Retrieval Evaluation



The saved retrieval sweep reported the following results for `top\_k = 8`:



| Metric | Result |

|---|---:|

| Section recall | 98.70% |

| Section precision | 20.99% |

| Questions with all expected sections retrieved | 46 / 47 (97.87%) |



These are retrieval metrics, not a measure of end-to-end answer correctness. Retrieving more evidence can improve coverage while also introducing irrelevant sections.



The repository includes additional baseline, RAG, comparison, and offline audit artifacts. Some saved answer evaluations were generated before the latest retrieval tuning, so those older answer results should be interpreted as historical snapshots rather than results from a fresh evaluation of the current retriever.



\## Evaluation Artifacts



The `outputs/` directory includes artifacts such as:



\- Baseline chatbot responses.

\- RAG chatbot responses.

\- Step 1 and Step 2 comparisons.

\- Offline answer-quality audit results.

\- Retrieval evaluation results.

\- Retrieval index files.

\- Sample voice input audio.



The project also includes evaluation scripts under `app/`.



LLM-based evaluation scripts can consume API quota or incur usage costs. Run them deliberately and review their configured scope before executing large evaluations.



\## Security and Data Handling



\- Never commit real API keys or other secrets.

\- Keep `.env` excluded from Git.

\- Use `.env.example` only for placeholder configuration.

\- The supplied ZentroMart data is synthetic.

\- Order lookups and ticket creation use local mock tools and CSV data.

\- Generated voice replies and runtime state should remain local unless explicitly required for submission.

\- Do not present the mock system as connected to a live customer database or real support team.



\## Limitations



\- Results depend on the selected LLM provider, model availability, and API quotas.

\- The RAG system can retrieve irrelevant evidence or miss relevant knowledge-base sections.

\- Saved evaluation outputs may represent earlier code or configuration versions.

\- The order and ticket tools operate on the supplied synthetic dataset.

\- The voice interface processes an audio file; the included workflow does not by itself establish a live telephone or real-time microphone deployment.

\- A successful sample voice run does not guarantee accurate transcription for every language, accent, or audio quality.

\- The application has not been established as a production deployment.



\## Repository



GitHub: https://github.com/saibalaji2004/voiceops-2026-capstone



\---



Developed as a VoiceOps 2026 capstone project demonstrating RAG, agentic customer support, verification-controlled tool use, and voice integration.

