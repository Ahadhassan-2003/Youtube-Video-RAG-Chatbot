# YouTube Video RAG Chatbot

A conversational AI chatbot that allows users to interact with YouTube video content using Retrieval-Augmented Generation (RAG). The application extracts video transcripts, creates a searchable knowledge base, and provides intelligent responses based on the video content.

## Live Demo

Try the application directly in your browser: [YouTube RAG Chatbot](https://ahadhassan-2003-youtube-video-rag-chatbot-app-c39c8m.streamlit.app/)

## Features

- **Video Transcript Processing**: Automatically extracts transcripts from YouTube videos
- **Conversational Interface**: Maintains conversation history for contextual responses
- **RAG Implementation**: Uses vector similarity search to find relevant content
- **Flexible Input**: Accepts both YouTube URLs and video IDs
- **Real-time Chat**: Interactive Streamlit interface with persistent chat history
- **Secure API Integration**: Protected OpenAI API key input

## Technology Stack

- **Frontend**: Streamlit
- **Language Model**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: FAISS
- **Framework**: LangChain
- **Video Processing**: YouTube Transcript API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

2. Install required dependencies:
```bash
pip install streamlit youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Setup**:
   - Enter your OpenAI API key in the sidebar
   - Input a YouTube video URL or video ID

2. **Processing**:
   - Click "Process Video" to extract and analyze the transcript
   - The system will create embeddings and set up the RAG pipeline

3. **Chat**:
   - Ask questions about the video content
   - Use follow-up questions for deeper discussions
   - Clear chat history when needed

### Example Interactions

```
User: What is this video about?
Assistant: This video discusses [summary based on transcript content]

User: Can you explain that in simpler terms?
Assistant: [Simplified explanation considering previous context]

User: What examples were mentioned?
Assistant: [Specific examples from the video transcript]
```

## How It Works

### 1. Document Ingestion
- Extracts video transcript using YouTube Transcript API
- Splits transcript into manageable chunks using RecursiveCharacterTextSplitter

### 2. Vector Store Creation
- Generates embeddings for text chunks using OpenAI embeddings
- Stores embeddings in FAISS vector database for fast similarity search

### 3. Retrieval Process
- User queries are embedded and compared against stored vectors
- Top-k most similar chunks are retrieved as context

### 4. Response Generation
- Combines retrieved context with conversation history
- Uses OpenAI's GPT model to generate contextually aware responses
- Maintains conversational flow with LangChain's message types

## Project Structure

```
youtube-rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── chatbot.ipynb         # Original Jupyter notebook implementation
```

## Key Components

### RAG Pipeline
- **Indexing**: Document processing and vector store creation
- **Retrieval**: Similarity search for relevant content
- **Augmentation**: Context preparation with conversation history
- **Generation**: Response creation using language model

### Conversation Management
- Uses LangChain's `HumanMessage` and `AIMessage` for proper message handling
- Maintains conversation history for contextual responses
- Implements token management to prevent context overflow

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (input through UI)

### Model Settings
- **LLM**: GPT-4o-mini with temperature 0.2
- **Embeddings**: text-embedding-3-small
- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top 4 most similar chunks

## Limitations

- Only works with YouTube videos that have available transcripts/captions
- Responses are limited to content within the video transcript
- Requires valid OpenAI API key with sufficient credits
- Chat history is limited to last 10 messages to manage token usage

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid API keys
- YouTube videos without transcripts
- Network connectivity issues
- OpenAI API rate limits
- Invalid video URLs or IDs

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls and video transcript fetching

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language models and embeddings
- LangChain for the RAG framework
- Streamlit for the web interface
- YouTube Transcript API for video content extraction
