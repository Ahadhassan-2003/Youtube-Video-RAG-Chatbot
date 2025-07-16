import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import re

# Page configuration
st.set_page_config(
    page_title="YouTube Video Chat",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        color: black;
    }
    .stTextInput input::placeholder {
        color: black;
        opacity: 0.3; /* Ensures it's fully visible */
    }
</style>
""", unsafe_allow_html=True)

def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return ID if already provided"""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        # Extract ID from URL
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:v\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
    elif len(url_or_id) == 11:
        # Assume it's already a video ID
        return url_or_id
    return None

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, "No captions available for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

def create_rag_chain(transcript, api_key):
    """Create RAG chain from transcript with conversation history support"""
    try:
        # Set API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Create chat prompt template with conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about a YouTube video.
            Answer ONLY from the provided transcript context and conversation history.
            If the context is insufficient, just say you don't know.
            Be conversational and helpful in your responses.
            You can reference previous parts of the conversation when relevant.
            
            Context from video transcript:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create chain that includes conversation history
        def create_chain_with_history(input_dict):
            # Get the current question for retrieval
            question = input_dict["question"]
            chat_history = input_dict.get("chat_history", [])
            
            # Retrieve relevant documents based on current question
            retrieved_docs = retriever.invoke(question)
            context = format_docs(retrieved_docs)
            
            # Format the prompt with context and history
            formatted_prompt = prompt.format_messages(
                context=context,
                chat_history=chat_history,
                question=question
            )
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            return response.content
        
        chain = RunnableLambda(create_chain_with_history)
        
        return chain, None
    except Exception as e:
        return None, f"Error creating RAG chain: {str(e)}"

def main():
    st.markdown("<h1 class='main-header'>🎥 YouTube Video Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key"
        )
        
        # Video ID input
        video_input = st.text_input(
            "YouTube Video URL or ID",
            placeholder="https://www.youtube.com/watch?v=... or video_id",
            help="Enter YouTube video URL or just the video ID"
        )
        
        # Process button
        process_button = st.button("🔄 Process Video", type="primary")
        
        if process_button:
            if not api_key:
                st.error("Please enter your OpenAI API key")
                return
            
            if not video_input:
                st.error("Please enter a YouTube video URL or ID")
                return
            
            # Extract video ID
            video_id = extract_video_id(video_input)
            if not video_id:
                st.error("Invalid YouTube URL or video ID")
                return
            
            # Store in session state
            st.session_state.video_id = video_id
            st.session_state.api_key = api_key
            st.session_state.processing = True
    
    # Main content area
    if hasattr(st.session_state, 'processing') and st.session_state.processing:
        with st.spinner("Processing video transcript..."):
            # Get transcript
            transcript, error = get_video_transcript(st.session_state.video_id)
            
            if error:
                st.error(error)
                st.session_state.processing = False
                return
            
            # Create RAG chain
            chain, error = create_rag_chain(transcript, st.session_state.api_key)
            
            if error:
                st.error(error)
                st.session_state.processing = False
                return
            
            # Store in session state
            st.session_state.chain = chain
            st.session_state.transcript = transcript
            st.session_state.processing = False
            st.session_state.ready = True
            st.success("Video processed successfully! You can now chat about the video.")
    
    # Chat interface
    if hasattr(st.session_state, 'ready') and st.session_state.ready:
        st.header("💬 Chat about the Video")
        
        # Initialize chat history with LangChain message types
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "langchain_history" not in st.session_state:
            st.session_state.langchain_history = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from RAG chain with conversation history
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Prepare input with conversation history
                        chain_input = {
                            "question": prompt,
                            "chat_history": st.session_state.langchain_history
                        }
                        response = st.session_state.chain.invoke(chain_input)
                    st.markdown(response)
                
                # Add messages to both display and LangChain history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.langchain_history.append(HumanMessage(content=prompt))
                st.session_state.langchain_history.append(AIMessage(content=response))
                
                # Keep only last 10 messages in history to avoid token limit
                if len(st.session_state.langchain_history) > 10:
                    st.session_state.langchain_history = st.session_state.langchain_history[-10:]
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.langchain_history = []
            st.rerun()
        
        # Video info
        with st.expander("📊 Video Information"):
            if hasattr(st.session_state, 'transcript'):
                st.write(f"**Video ID:** {st.session_state.video_id}")
                st.write(f"**Transcript Length:** {len(st.session_state.transcript)} characters")
                st.write(f"**YouTube Link:** https://www.youtube.com/watch?v={st.session_state.video_id}")
    
    else:
        # Instructions
        st.markdown("""
        ## How to use:
        
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Paste a YouTube video URL or ID** in the sidebar
        3. **Click "Process Video"** to analyze the video transcript
        4. **Start chatting** about the video content!
        
        ### Example questions you can ask:
        - "What is this video about?"
        - "Can you summarize the main points?"
        - "What does the speaker say about [specific topic]?"
        - "Are there any examples mentioned?"
        
        ### Note:
        - Only videos with available captions/transcripts can be processed
        - The chatbot will only answer based on the video content
        - Make sure your OpenAI API key is valid and has sufficient credits
        """)

if __name__ == "__main__":
    main()