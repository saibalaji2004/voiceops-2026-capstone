# VoiceOps 2026 Capstone

### Agentic AI-Based Voice Customer Support System

VoiceOps is an AI-powered customer support assistant developed as a capstone project using a synthetic ZentroMart customer support dataset. The application combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), conversation memory, tool-based order verification, support ticket creation, and speech processing.

The project demonstrates the development of a customer support agent through four stages: a prompt-only baseline, a RAG-based assistant, a stateful agent with tools, and a voice-enabled interface.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Four Implementation Stages](#four-implementation-stages)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Configuration](#installation-and-configuration)
- [Running the Application](#running-the-application)
- [Batch Audio Processing](#batch-audio-processing)
- [Validation and Evaluation](#validation-and-evaluation)
- [Evaluation Artifacts](#evaluation-artifacts)
- [Security and Data Handling](#security-and-data-handling)
- [Limitations](#limitations)
- [Submission Package](#submission-package)
- [Repository](#repository)

## Project Overview

Traditional chatbots often struggle to maintain context, retrieve the correct policy information, verify customer identity, and perform customer support actions safely.

VoiceOps explores how an AI customer support system can address these challenges by combining several components:

- LLM-based response generation.
- Retrieval of relevant knowledge-base information.
- Conversation history and follow-up handling.
- Controlled access to mock order lookup and support ticket tools.
- Caller verification before disclosing protected order information.
- Speech-to-text (STT) and text-to-speech (TTS) integration.

The project uses a supplied synthetic dataset containing customer records, orders, products, policies, knowledge-base documents, practice questions, and conversation examples.

## Key Features

### 1. Prompt-Based Customer Support

A baseline chatbot generates responses using instructions and an LLM without the main RAG retrieval workflow.

### 2. Retrieval-Augmented Generation

The RAG implementation retrieves relevant knowledge-base sections and product information to provide evidence-grounded customer support responses.

The retrieval evaluation includes comparisons across different values of `top_k`.

### 3. Stateful Agentic Customer Support

The agent implementation supports:

- Conversation history and follow-up questions.
- Order ID collection and order lookup.
- Customer verification using the supported verification methods.
- Restrictions on order disclosures before successful verification.
- Limited verification retries.
- Support ticket creation through the supplied mock tool.
- Controlled handover messaging based on tool outcomes.

Order operations use the local synthetic dataset and mock tools, not a live commercial order-management system.

### 4. Voice Interface

The voice module integrates Gnani speech services to:

1. Transcribe an input audio file into text.

2. Send the transcribed message to the customer support agent.

3. Generate a spoken response using text-to-speech.

4. Save the generated audio response to a file.

The English (India) voice workflow was tested successfully using an audio file.

## Four Implementation Stages

| Stage | Implementation | Purpose |
|---|---|---|
| Step 1 | `app/chatbot.py` | Establish a prompt-only baseline |
| Step 2 | `app/rag.py`, `app/rag_chatbot.py` | Retrieve relevant evidence for customer responses |
| Step 3 | `app/step3_agent.py` | Maintain conversation state and use support tools safely |
| Step 4 | `app/voice.py` | Integrate speech-to-text and text-to-speech |

Supporting modules include the LLM provider router, evaluation scripts, retrieval audits, and the interactive application entry point.

## Technology Stack

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

## System Architecture

```
                 Customer Message
                        |
                        v
              Interactive Application
                        |
                        v
                Stateful Agent
                        |
           +------------+-------------+
           |                          |
           v                          v
      Conversation                 RAG Retrieval
       History                    and Knowledge Base
           |                          |
           +------------+-------------+
                        |
                        v
               LLM Response Router
                        |
                        v
               Customer Response
                        |
            +-----------+-----------+
            |                       |
            v                       v
       Order Lookup             Ticket Creation
       and Verification         and Handover
            |
            v
       Synthetic Dataset
Audio Input --> Gnani STT --> Agent --> Gnani TTS --> Audio Output
```

## Project Structure

The main repository contains the application and helper scripts. The supplied dataset is downloaded separately and is not part of the submission ZIP.

```text
voiceops-2026-capstone/
├── app/
│   ├── chatbot.py
│   ├── config.py
│   ├── llm_router.py
│   ├── main.py
│   ├── rag.py
│   ├── rag_chatbot.py
│   ├── step3_agent.py
│   ├── voice.py
│   ├── evaluate_step1.py
│   ├── evaluate_step2.py
│   ├── evaluate_retrieval_k.py
│   ├── compare_evaluations.py
│   └── audit_step2_retrieval.py
├── outputs/                 # Local evaluation/voice artifacts; may be generated
├── sample_outputs/           # Example batch outputs included in submission ZIP
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py                    # Folder-based audio runner
└── README.md
```

The separately extracted `Capstone_Project_Dataset_FTTT_2026/` folder must be present locally for dataset-backed lookups and validation. Do not include it in the submission ZIP. The `sample_outputs/` folder contains representative runner results for review; additional files may be generated during execution.

## Prerequisites

- Windows, macOS, or Linux with an appropriate Python environment.
- Python 3.11 recommended for the tested environment.
- Git for source control.
- API credentials for the LLM provider or providers you intend to use.
- Gnani API credentials for the voice workflow.
- Internet access for the configured cloud-based model and speech services.

## Installation and Configuration

### 1. Clone the repository

```
git clone https://github.com/saibalaji2004/voiceops-2026-capstone.git
cd voiceops-2026-capstone
```

### Obtain the supplied dataset separately

The official `Capstone_Project_Dataset_FTTT_2026` dataset is provided separately in the capstone materials. Download and extract it into the repository root so the application can locate the data and knowledge-base files at `Capstone_Project_Dataset_FTTT_2026/`.

Read the dataset's `README.md` and `DATA_DICTIONARY.md` before running the application. Do not place the dataset ZIP or extracted dataset folder in the submission ZIP. Keep the supplied dataset out of the public repository if the capstone instructions prohibit redistribution.

### 2. Create a virtual environment

On Windows PowerShell:

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use an appropriate environment activation method for your machine rather than changing system-wide security settings unnecessarily.

### 3. Install dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from the provided template:

```
Copy-Item .env.example .env
```

Run this copy command only when `.env` does not already exist. If you have already configured your credentials, do not overwrite your existing file.

Open `.env` and configure the credentials required by your selected providers. The template contains placeholder values for the available LLM providers and Gnani.

Keep actual credentials in `.env`, not in `README.md`, source code, screenshots, or `.env.example`.

## Running the Application

### Start the interactive customer support agent

From the project root:

```powershell
python -m app.main
```

Enter customer messages in the terminal and follow the prompts. The exact response depends on the supplied question, retrieved evidence, configured providers, and agent state.

### Run the voice interface

Use the supplied sample audio file if you have it available:

```powershell
python -m app.voice "outputs\voice_test.wav" --language en-IN
```

The command transcribes the audio, processes the message through the agent, and saves the generated speech response to `outputs/voice_reply.wav`. This requires valid Gnani credentials and a compatible audio input.

## Batch Audio Processing

The required folder-based runner is `run.py`. It processes each supported audio file in an input folder and writes the transcript/reply text, generated speech audio, retrieved evidence, and a batch manifest to the output folder.

Place input recordings in a folder such as `customer_audio/`, then run this single command from the repository root:

```powershell
python run.py --input-dir customer_audio --output-dir sample_outputs --language en-IN
```

For multiple turns that belong to the same conversation, enable shared conversation state:

```powershell
python run.py --input-dir customer_audio --output-dir sample_outputs --language en-IN --shared-session
```

The runner supports recursive folder scanning when requested:

```powershell
python run.py --input-dir customer_audio --output-dir sample_outputs --language en-IN --recursive
```

Use `--help` to see all available options:

```powershell
python run.py --help
```

Supported audio formats include WAV, MP3, M4A, FLAC, OGG, and WEBM. Output filenames are derived from the input filenames. Keep input recordings in a separate folder from the output folder to avoid processing generated files again. The voice workflow requires valid Gnani credentials and a configured LLM provider.

### Validate the supplied dataset

```
python Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
```

### Compile-check the Python modules

```
python -m compileall -q app Capstone_Project_Dataset_FTTT_2026\tools
```

## Validation and Evaluation

The following checks were completed in the development environment:

- Python compilation completed successfully.
- The supplied dataset validator passed.
- The dataset validation covered all 9 CSV files, 21 knowledge-base documents, primary and foreign keys, order calculations, practice evidence, verification gates, and company naming.
- The validated dataset contains 1,000 orders, 1,783 order items, 244 support logs, 52 practice questions, and 23 conversations.
- The order lookup and caller verification workflow was exercised successfully.
- Support ticket creation was exercised using the mock tool.
- The Gnani STT-to-agent-to-TTS workflow completed successfully for sample English (India) audio.
- The folder-based `run.py` workflow was exercised on a small set of audio files, producing text, audio, manifest, and source-evidence artifacts.
- A recorded multi-turn demonstration was processed through the shared-session batch workflow; the sample follow-up response was checked after a prompt adjustment to reduce repeated text.

These results describe the checks performed in the development environment. They do not establish that every possible conversation has been tested or that the full 23-conversation / 96-turn practice evaluation is complete.

### Retrieval Evaluation

The saved retrieval sweep reported the following results for `top_k = 8`:

| Metric | Result |
|---|---:|
| Section recall | 98.70% |
| Section precision | 20.99% |
| Questions with all expected sections retrieved | 46 / 47 (97.87%) |

These are retrieval metrics, not a measure of end-to-end answer correctness. Retrieving more evidence can improve coverage while also introducing irrelevant sections.

The repository includes additional baseline, RAG, comparison, and offline audit artifacts. Some saved answer evaluations were generated before the latest retrieval tuning, so those older answer results should be interpreted as historical snapshots rather than results from a fresh evaluation of the current retriever.

## Evaluation Artifacts

The `outputs/` directory includes artifacts such as:

- Baseline chatbot responses.
- RAG chatbot responses.
- Step 1 and Step 2 comparisons.
- Offline answer-quality audit results.
- Retrieval evaluation results.
- Retrieval index files.
- Sample voice input audio.

The project also includes evaluation scripts under `app/`.

LLM-based evaluation scripts can consume API quota or incur usage costs. Run them deliberately and review their configured scope before executing large evaluations.

## Security and Data Handling

- Never commit real API keys or other secrets.
- Keep `.env` excluded from Git.
- Use `.env.example` only for placeholder configuration.
- The supplied ZentroMart data is synthetic.
- Order lookups and ticket creation use local mock tools and CSV data.
- Generated voice replies and runtime state should remain local unless explicitly required for submission.
- Do not present the mock system as connected to a live customer database or real support team.

## Limitations

- Results depend on the selected LLM provider, model availability, and API quotas.
- The RAG system can retrieve irrelevant evidence or miss relevant knowledge-base sections.
- Saved evaluation outputs may represent earlier code or configuration versions.
- The order and ticket tools operate on the supplied synthetic dataset.
- The voice interface processes an audio file; the included workflow does not by itself establish a live telephone or real-time microphone deployment.
- A successful sample voice run does not guarantee accurate transcription for every language, accent, or audio quality.
- The application has not been established as a production deployment.

## Submission Package

The capstone submission ZIP should be named `team_name_voiceops.zip` and contain the required project files, including:

- `README.md`
- `.env.example` with placeholder values only
- `requirements.txt`
- `run.py`
- `app/` source code
- `report.pdf` (maximum four pages)
- `demo.mp4` (maximum two minutes)
- `sample_outputs/` with representative text replies, generated audio replies, source-evidence JSON files, and the batch manifest

Do not include `.env`, API keys, `.venv/`, cache folders, or the supplied dataset. The submission ZIP is separate from the GitHub repository push; upload the ZIP to the official capstone submission portal and verify that the portal shows a successful submission/confirmation.

## Repository

GitHub: [https://github.com/saibalaji2004/voiceops-2026-capstone](https://github.com/saibalaji2004/voiceops-2026-capstone)

---

Developed as a VoiceOps 2026 capstone project demonstrating RAG, agentic customer support, verification-controlled tool use, and voice integration.
