import streamlit as st
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from tavily import TavilyClient
import json
from typing import List
from utils import get_containerstyle
from datetime import datetime


# Initialize clients
oaiClient = OpenAI(api_key=st.secrets.openai.aether_api_key)
tavClient = TavilyClient(api_key=st.secrets.tavily.api_key)

# Set AI Avatar
ai_avatar = st.secrets.app.chat_icon



# Get current date dynamically
current_date = datetime.now().strftime("%m/%d/%Y")

# Define the prompt with a placeholder for the date
prompt = """
You are a wrestling expert, specializing in youth, high school, collegiate, and Olympic wrestling, including folkstyle, freestyle, and Greco-Roman styles. Assist users with their wrestling-related inquiries using your extensive knowledge of techniques, rules, philosophies, and strategies.

IMPORTANT - THE CURRENT DATE IS {date} - YOU WILL USE THIS IN YOUR SEARCH QUERY.

# Guidelines

- Only provide information related to amateur wrestling; do not address fake wrestling.
- For current events, perform a web search including "amateur wrestling."
- Use "https://themat.com", "https://uww.org" and "https://flowrestling.com" with the Extract Content tool to find relevant information. DO NOT USE SEARCH WEB TOOL WITH THESE
- Assume no current events knowledge without a search.

# Output Format

Provide clear and informative responses tailored to the user's specific wrestling-related query.
"""

# Format the prompt with the current date
formatted_prompt = prompt.format(date=current_date)

# Tool Json
web_search_json = {
        "name": "web_search",
        "description": "Search the web with a given query and return relevant results.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
            "query"
            ],
            "properties": {
            "query": {
                "type": "string",
                "description": "The search query string"
            }
            },
            "additionalProperties": False
        }
        }


extract_content_json ={
        "name": "extract_content",
        "description": "Extract content from specific URLs.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
            "urls"
            ],
            "properties": {
            "urls": {
                "type": "array",
                "description": "List of URLs to extract content from",
                "items": {
                "type": "string",
                "description": "A URL to extract content from"
                }
            }
            },
            "additionalProperties": False
        }
        }


# Tool functions
def web_search(query: str): 
    response = tavClient.search(query=query, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=True)
    return response

def extract_content(urls: List[str]):
    response = tavClient.extract(urls=urls, include_images=False)
    return response

class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.text_buffer = []  # Buffer to collect text deltas for streaming

    @override
    def on_text_created(self, text) -> None:
        """Handle the creation of new text"""
        st.toast("Assistant started responding...")

    @override
    def on_text_delta(self, delta, snapshot):
        """Handle text deltas as they stream in"""
        if delta.value:
            self.text_buffer.append(delta.value)

    @override
    def on_event(self, event):
        """Handle all events"""
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self.handle_requires_action(event.data, run_id)
        elif event.event == "thread.run.failed":
            st.error(f"Run failed: {event.data.last_error}")

    def handle_requires_action(self, data, run_id):
        """Handle tool calls when action is required"""
        tool_outputs = []
        for tool in data.required_action.submit_tool_outputs.tool_calls:
            tname = tool.function.name
            targs = json.loads(tool.function.arguments)
            tid = tool.id
            try:
                if tname == "web_search":
                    tresult = web_search(query=targs['query'])
                elif tname == "extract_content": 
                    tresult = extract_content(urls=targs['urls'])
                else:
                    tresult = f"Unknown tool: {tname}"
                tool_outputs.append({"tool_call_id": tid, "output": json.dumps(tresult)})
            except Exception as e:
                tool_outputs.append({"tool_call_id": tid, "output": f"Error: {str(e)}"})
        
        if tool_outputs:
            self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        """Submit tool outputs and stream the response with a new handler"""
        new_handler = EventHandler()
        with oaiClient.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
            event_handler=new_handler
        ) as stream:
            stream.until_done()
        # Extend the buffer with the new handler's results
        self.text_buffer.extend(new_handler.text_buffer)

    def generate_text(self):
        """Generator to yield text chunks"""
        for chunk in self.text_buffer:
            yield chunk

def setup_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            'role': 'assistant', 
            'content': 'Welcome to AetherAI - how can I assist you today?'
        }]
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = oaiClient.beta.threads.create().id
    
    if "assistant_id" not in st.session_state:
        st.session_state.assistant = oaiClient.beta.assistants.create(instructions=formatted_prompt, name="Aether Assistant - Custom1", model="gpt-4o",tools=[{"type": "file_search"}, {"type": "function", "function": extract_content_json}, {"type": "function", "function": web_search_json}],tool_resources={"file_search": {"vector_store_ids": [st.secrets.openai.aether_vectorstore_id]}})
        st.session_state.assistant_id = st.session_state.assistant.id

def display_chat_history(chat_container):
    """Display all messages in the chat history within the provided container"""
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == "assistant":
                avatar = ai_avatar
            else:
                avatar = None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

def process_user_input_streaming(assistant_id, chat_container, prompt_container):
    """Handle user input and stream assistant response with tool calls"""
    # Display chat input in the prompt container
    with prompt_container:
        prompt = st.chat_input(placeholder="Type here...")
    
    if prompt:
        # Add user message to session state and display in chat container
        user_message = {'role': 'user', 'content': prompt}
        st.session_state.messages.append(user_message)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Add message to thread
        oaiClient.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            content=prompt,
            role="user"
        )

        # Create run with streaming enabled and display in chat container
        with chat_container:
            with st.chat_message("assistant", avatar=ai_avatar):
                handler = EventHandler()
                response_placeholder = st.empty()  # Create a placeholder for streaming response
                full_response = ""
                with oaiClient.beta.threads.runs.stream(
                    thread_id=st.session_state.thread_id,
                    assistant_id=assistant_id,
                    event_handler=handler
                ) as stream:
                    stream.until_done()
                    # Stream the response incrementally as deltas arrive
                    for chunk in handler.generate_text():
                        full_response += chunk
                        response_placeholder.markdown(full_response)
        
        # After streaming completes, add to session state
        if full_response:
            assistant_message = {'role': 'assistant', 'content': full_response}
            st.session_state.messages.append(assistant_message)

def main():
    """Main application function"""
    #assistant_id = st.secrets.openai.aether_assistant_id
    
    # Create containers
    #chat_container = st.container(border=True, height=200)
    chat_container = get_containerstyle(height=200, border=False)
    prompt_container = st.container(border=False, height=50)
    # chat_container = st.container(border = False)
    # prompt_container = st.container(border=False)
    #chat_container = st.container(border=True, height=400)
    #prompt_container = st.container(border=False, height=200)
    
    setup_session_state()
    display_chat_history(chat_container)
    process_user_input_streaming(st.session_state.assistant_id, chat_container, prompt_container)

if __name__ == "__main__":
    main()
