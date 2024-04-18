# LLM-For-Websites

This is a Streamlit-based application that allows users to chat with a chatbot powered by a large language model (LLM) and retrieve information from a website.

## Features

- Loads website content and creates a vector store using LangChain
- Provides a chat interface for users to interact with the chatbot
- Uses a retrieval-augmented generation (RAG) chain to generate responses based on the website content and chat history
- Supports two LLM options:
  - **Mistral** (from GPT4All): This is an open-source model that can be used locally, but it is slower than the OpenAI option.
  - **OpenAI**: This option uses the OpenAI API and provides faster response times, but requires an API key to be set in a `.env` file.

## Prerequisites

1. Python 3.7 or higher
2. The required Python packages: `streamlit`, `langchain`, `langchain-openai`, `beautifulsoup4`, `python-dotenv`, `chromadb`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/llm-website-chatbot.git
   ```

2. Change to the project directory:

   ```
   cd llm-website-chatbot
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download the Mistral model from the [GPT4All website](https://www.gpt4all.io/models.html) and save it to the `/models` directory in your project.

2. Update the `local_path` variable in the `get_context_retriever_chain` and `get_conversational_rag_chain` functions to point to the downloaded Mistral model file.

   ```python
   llm = GPT4All(model="/models/mistral-7b-openorca.gguf2.Q4_0.gguf")
   ```

3. If you want to use the OpenAI option, follow these steps:

   - Create a `.env` file in the project directory.
   - Add your OpenAI API key to the `.env` file:

     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

   - In the code, uncomment the `load_dotenv()` line and replace the `GPT4All` instances with `ChatOpenAI` and `OpenAIEmbeddings`.

4. Run the Streamlit application:

   ```
   streamlit run app.py
   ```

5. The application will open in your default web browser. Enter a website URL in the sidebar, and start chatting with the chatbot.

## Deployment

This application is designed to run locally, as it makes use of the open-source Mistral model from GPT4All. To deploy the application, you would need to:

1. Host the application on a web server or cloud platform (e.g., Heroku, AWS, GCP).
2. Ensure that the Mistral model file is accessible to the deployed application.
3. If using the OpenAI option, make sure to securely store the OpenAI API key in the deployment environment.

## Acknowledgments

This project was built using the LangChain library, which provides a powerful set of tools for building applications with large language models. The Mistral model is from the GPT4All project, an open-source ecosystem of chatbots.
