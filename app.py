import streamlit as st
import openai
import os
import asyncio
from dotenv import load_dotenv
import http.client # Use standard http.client for Serper.dev
import json # Use standard json library
import httpx # Required by OpenAI library, keep it

# --- Configuration and Setup ---

# Load credentials from .env file
load_dotenv("credentials.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Basic validation for API keys
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in credentials.env. Please add it.")
    st.stop() # Stop execution if key is missing
if not SERPER_API_KEY:
    st.error("❌ SERPER_API_KEY not found in credentials.env. Please add it.")
    st.stop() # Stop execution if key is missing

# Initialize OpenAI client (Async)
# Streamlit generally works well with top-level async functions
try:
    openai_client = openai.AsyncOpenAI()
except openai.OpenAIError as e:
    st.error(f"❌ Error initializing OpenAI client: {e}")
    st.stop()

# --- Helper Functions ---

async def perform_serper_search(query: str, location: str | None = None) -> str:
    """
    Performs a web search using Serper.dev and returns formatted results.
    Uses http.client for the request.
    """
    if not SERPER_API_KEY:
        return "Web search is not configured (missing Serper API key)."

    print(f"Performing Serper.dev search for: '{query}'" + (f" near '{location}'" if location else ""))
    search_results_str = "No relevant web search results found."

    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload_dict = {
            "q": query,
            "gl": "sa", # Country code for Saudi Arabia
            "hl": "en", # Language
            "num": 7 # Number of results
        }
        if location:
            payload_dict["location"] = location
            # For local search, specify the type if needed, e.g., 'psychiatrists'
            # payload_dict['type'] = 'search' # or 'places' depending on what works best

        payload = json.dumps(payload_dict)
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        # Run synchronous http.client call in a separate thread
        def make_request():
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            status = res.status
            reason = res.reason
            conn.close()
            if status >= 400:
                 # Raise an exception for bad status codes
                 raise httpx.HTTPStatusError(f"Serper API error: {status} {reason}", request=None, response=res) # Simulate httpx error structure
            return data.decode("utf-8")

        response_data_str = await asyncio.to_thread(make_request)
        results = json.loads(response_data_str)

        # --- Parse Serper.dev results ---
        organic_results = results.get("organic", [])
        local_results = results.get("places", []) # Serper often uses 'places' for local results
        answer_box = results.get("answerBox")
        knowledge_graph = results.get("knowledgeGraph")

        formatted_results = []

        # Add knowledge graph info if available
        if knowledge_graph:
            kg_title = knowledge_graph.get("title", "Info")
            kg_type = knowledge_graph.get("type", "")
            kg_desc = knowledge_graph.get("description", "")
            kg_url = knowledge_graph.get("url")
            formatted_results.append(f"**ℹ️ {kg_title} ({kg_type})**")
            if kg_desc: formatted_results.append(f"> {kg_desc}")
            if kg_url: formatted_results.append(f"> Source: {kg_url}")
            formatted_results.append("---") # Separator

        # Add answer box info if available
        if answer_box:
            ab_snippet = answer_box.get("snippet") or answer_box.get("answer")
            ab_title = answer_box.get("title")
            ab_link = answer_box.get("link")
            if ab_snippet:
                 formatted_results.append("**Quick Answer:**")
                 formatted_results.append(f"> {ab_snippet}")
                 if ab_title and ab_link: formatted_results.append(f"> Source: [{ab_title}]({ab_link})")
                 formatted_results.append("---") # Separator


        # Prioritize local results if location was specified and results exist
        if location and local_results:
            formatted_results.append("**📍 Local Psychiatrists Found:**")
            for i, result in enumerate(local_results[:5]): # Show top 5 local results
                name = result.get("title", "N/A")
                address = result.get("address", "N/A")
                phone = result.get("phone")
                website = result.get("website")
                rating = result.get("rating", "N/A")
                reviews = result.get("reviewCount", "N/A") # Serper uses reviewCount
                category = result.get("category", "N/A")

                entry = f"**{i+1}. {name}** ({category})\n"
                if address: entry += f"   Address: {address}\n"
                if phone: entry += f"   Phone: {phone}\n"
                if website: entry += f"   Website: <{website}>\n" # Use angle brackets for links in markdown
                entry += f"   Rating: {rating} ({reviews} reviews)"
                formatted_results.append(entry)
            formatted_results.append("---") # Separator


        # Add organic results (limit if local results were shown)
        num_organic_to_show = 3 if (location and local_results) else 5
        if organic_results:
             if not (location and local_results): # Only add header if no local results shown
                 formatted_results.append("**🌐 Web Search Results:**")

             for i, result in enumerate(organic_results[:num_organic_to_show]):
                title = result.get("title", "No Title")
                link = result.get("link", "#")
                snippet = result.get("snippet", "No snippet available.")
                formatted_results.append(f"**{i+1}. {title}**\n   Snippet: {snippet}\n   Source: <{link}>")

        if formatted_results:
            search_results_str = "\n\n".join(formatted_results)
            print(f"Serper.dev search successful.")
        else:
            print("Serper.dev search completed, but no relevant results found.")
            search_results_str = "I couldn't find specific results for that query."

    except httpx.HTTPStatusError as e: # Catch potential error from make_request
        print(f"Serper API HTTP error: {e}")
        search_results_str = f"An error occurred during the web search (API issue: {e}). Please check the Serper API key and usage limits."
    except json.JSONDecodeError as e:
        print(f"Error decoding Serper.dev response: {e}")
        search_results_str = "An error occurred processing search results."
    except Exception as e:
        print(f"Error during Serper.dev search: {e}")
        search_results_str = f"An unexpected error occurred during the web search: {e}"

    return search_results_str

async def search_local_psychiatrists(location: str, symptoms: str) -> str:
    """
    Wrapper function to perform a localized psychiatrist search using Serper.dev.
    """
    if not location:
        return "⚠️ Please provide a location to search for psychiatrists."

    query = f"psychiatrists specializing in {symptoms} near {location}" if symptoms else f"psychiatrists near {location}"
    return await perform_serper_search(query, location=location)

async def perform_general_web_search(query: str) -> str:
    """
    Wrapper function to perform a general web search using Serper.dev.
    """
    return await perform_serper_search(query, location=None)

async def get_openai_response_stream(message: str, history: list):
    """
    Gets a streaming response from OpenAI, incorporating history and web search.
    Yields chunks of text for Streamlit's st.write_stream.
    """
    print(f"Received message for OpenAI: {message}")
    # print(f"History for OpenAI: {history}") # Can be verbose

    # --- System Prompt ---
    system_prompt = """You are 'MindGuide', an empathetic and supportive AI assistant designed to provide helpful information and guidance in a psychiatric context.
    Your primary goal is to listen, understand, and offer supportive, general information.
    You MUST NOT provide medical advice, diagnosis, or treatment recommendations.
    You MUST encourage users to consult with qualified healthcare professionals for personal concerns.
    Prioritize safety. If a user expresses thoughts of self-harm or harming others, gently guide them towards professional help or emergency services immediately (e.g., "If you're feeling overwhelmed, reaching out to a crisis hotline or mental health professional can provide immediate support. Help is available.") and cease further interaction on the harmful topic.
    Be calm, patient, and non-judgmental. Use clear and simple language.
    If asked about recent events, specific treatments, medication details, or location-based resources (unless the user is using the dedicated 'Find Psychiatrist' feature), use the provided web search results to inform your answer. Cite search results naturally within your response when relevant (e.g., "According to [Source Title], ...").
    Keep responses concise and helpful. Avoid overly long paragraphs. Use markdown formatting like bolding and lists where appropriate for readability.
    """

    # --- Web Search Integration ---
    search_needed = any(keyword in message.lower() for keyword in ["latest", "recent", "news", "study", "treatment", "medication", "side effect", "statistic", "research on"]) \
                    and not any(loc_keyword in message.lower() for loc_keyword in ["near me", "in my area", "local", "clinic", "therapist", "psychiatrist"])

    search_context = ""
    if search_needed:
        search_results = await perform_general_web_search(message)
        if search_results and "No relevant" not in search_results and "not configured" not in search_results and "error occurred" not in search_results:
            search_context = f"\n\nWeb Search Context:\n---\n{search_results}\n---\nUse the above context ONLY IF RELEVANT to answer the user's question: '{message}'"
            print("Added general web search results to the prompt context.")

    # --- Construct Messages for OpenAI ---
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (ensure alternating user/assistant roles)
    for i, (role, content) in enumerate(history):
        if content: # Avoid adding empty messages
             messages.append({"role": role, "content": str(content)})

    # Add current user message, potentially augmented with search context
    user_content_with_context = message + search_context
    messages.append({"role": "user", "content": user_content_with_context})

    print(f"Sending {len(messages)} messages to OpenAI.")
    # print(f"Messages: {messages}") # Can be very verbose

    # --- Call OpenAI API (Streaming) ---
    try:
        stream = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
            temperature=0.7,
        )

        # Yield chunks directly for st.write_stream
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

        print("OpenAI stream finished.")

    except openai.APIConnectionError as e:
        print(f"OpenAI APIConnectionError: {e}")
        yield "Error: Sorry, I couldn't connect to the AI service. Please check the connection."
    except openai.RateLimitError as e:
        print(f"OpenAI RateLimitError: {e}")
        yield "Error: Sorry, the AI service is currently overloaded. Please try again later."
    except openai.APIStatusError as e:
        print(f"OpenAI APIStatusError: status={e.status_code}, response={e.response}")
        yield f"Error: Sorry, there was an issue with the AI service (Status: {e.status_code}). Please try again."
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI: {e}")
        yield "Error: Sorry, an unexpected error occurred while generating the response."

# --- Streamlit App ---

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="MindGuide AI",
    page_icon="🧠",
    layout="wide", # Use wide layout
    initial_sidebar_state="collapsed", # Start with sidebar collapsed
)

# Custom CSS for better appearance
st.markdown("""
<style>
    /* General Styling */
    .stApp {
        background-color: #f0f2f6; /* Light grey background */
    }
    /* Chat Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="chatAvatarIcon-user"] {
         background-color: #cce5ff; /* Light blue for user */
         color: #004085;
    }
     [data-testid="chatAvatarIcon-assistant"] {
         background-color: #d4edda; /* Light green for assistant */
         color: #155724;
    }
    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.5rem; /* Spacing between paragraphs */
        line-height: 1.6;
    }
    /* Input Area */
    [data-testid="stChatInput"] textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        background-color: #ffffff;
    }
    [data-testid="stChatInput"] button {
         border-radius: 8px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px; /* Space between tabs */
        background-color: #e9ecef;
        padding: 5px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        font-weight: bold;
        border-bottom: 2px solid #007bff; /* Highlight selected tab */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        color: white;
        background-color: #007bff; /* Primary button color */
        transition: background-color 0.3s ease, transform 0.1s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    .stButton>button:active {
        background-color: #004085;
        transform: scale(0.98);
    }
    /* Text Inputs */
    .stTextInput input, .stTextArea textarea {
         border-radius: 8px;
         border: 1px solid #ced4da;
    }
    /* Markdown for search results */
    [data-testid="stMarkdownContainer"] {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
     [data-testid="stMarkdownContainer"] h2 {
        margin-top: 0;
        color: #0056b3;
     }
     [data-testid="stMarkdownContainer"] strong {
        color: #004085;
     }
     [data-testid="stMarkdownContainer"] a {
        color: #007bff;
        text-decoration: none;
     }
     [data-testid="stMarkdownContainer"] a:hover {
        text-decoration: underline;
     }
     [data-testid="stMarkdownContainer"] hr {
         border-top: 1px solid #eee;
         margin: 1rem 0;
     }


</style>
""", unsafe_allow_html=True)

# --- App Layout ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# App Header
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://img.icons8.com/external-flaticons-lineal-color-flatart/100/external-brain-ai-flaticons-lineal-color-flatart.png", width=80) # Simple brain icon
with col2:
    st.title("🧠 MindGuide AI")
    st.caption("Your Empathetic AI Assistant for Information and Support")

st.markdown("---") # Separator

# --- Tabs for Features ---
tab1, tab2 = st.tabs(["💬 Chat with MindGuide", "🧑‍⚕️ Find Local Psychiatrists"])

# --- Chat Tab ---
with tab1:
    st.info("""
    Chat with MindGuide for supportive information. **Remember:** This AI cannot provide medical advice.
    For personal health concerns, please consult a qualified professional.
    If you are in crisis, contact emergency services.
    """, icon="ℹ️")

    # Display existing chat messages
    for role, content in st.session_state.messages:
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
            st.markdown(content) # Render content as Markdown

    # Chat input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message to history and display it
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Get and display AI response stream
        with st.chat_message("assistant", avatar="🧠"):
            # Use st.write_stream for real-time display
            response_stream = get_openai_response_stream(prompt, [(m[0], m[1]) for m in st.session_state.messages[:-1]]) # Pass history excluding the last user msg
            full_response = st.write_stream(response_stream)

        # Add the final AI response to the history
        st.session_state.messages.append(("assistant", full_response))
        # No need to rerun here, write_stream updates the UI

# --- Find Psychiatrist Tab ---
with tab2:
    st.subheader("Find Psychiatrists Near You")
    st.markdown("Enter your location and optionally symptoms or specialty needed.")

    with st.container(border=True): # Use a container for better visual grouping
        col_loc, col_symp = st.columns(2)
        with col_loc:
            user_location = st.text_input("Your City and State/Province*", placeholder="e.g., Jeddah, Makkah Province")
        with col_symp:
            user_symptoms = st.text_input("Primary Symptoms or Specialty Needed (Optional)", placeholder="e.g., anxiety, ADHD, child psychiatry")

        search_button = st.button("🔍 Find Psychiatrists", type="primary") # Use primary button style

    st.markdown("---") # Separator

    # Placeholder for search results
    results_placeholder = st.empty()

    if search_button:
        if not user_location:
            results_placeholder.warning("⚠️ Please enter your location.", icon="📍")
        else:
            with st.spinner("Searching for psychiatrists..."):
                # Call the search function
                search_results = asyncio.run(search_local_psychiatrists(user_location, user_symptoms))
                # Display results in the placeholder
                results_placeholder.markdown(search_results, unsafe_allow_html=True) # Allow HTML for links if needed

# --- Footer/Disclaimer ---
st.markdown("---")
st.caption("MindGuide AI - Always consult with a healthcare provider for diagnosis and treatment.")

