import streamlit as st
import streamlit.components.v1
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from typing import List
import re
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Text Humanizer RAG",
    page_icon="🤖➡️👨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TextHumanizerRAG:
    def __init__(self, api_key: str):
        """Initialize the RAG system with ChatGroq client"""
        self.api_key = api_key
        self.model = "gemma2-9b-it"
        self.llm = None
        self.temperature = 0.7
        self.update_model_settings(self.model, self.temperature)
        
    def update_model_settings(self, model: str, temperature: float):
        """Update model and temperature settings"""
        self.model = model
        self.temperature = temperature
        self.llm = ChatGroq(
            model=self.model,
            temperature=temperature,
            groq_api_key=self.api_key
        )
        
    def create_humanization_prompt(self, text: str, style: str = "conversational") -> str:
        """Create a prompt for humanizing text based on style preferences"""
        style_prompts = {
            "conversational": """Transform the following text to sound more conversational and human-like:
- Use natural speech patterns
- Add appropriate contractions (don't, won't, can't)
- Include casual transitions and connectors
- Make it sound like someone is actually talking
- Keep the core meaning intact""",

"professional": """Rewrite the following text to sound more professional yet human:
- Use polished but natural language
- Maintain professionalism while adding warmth
- Include appropriate business-friendly expressions
- Make it sound like an expert speaking naturally
- Keep the technical accuracy intact""",

"friendly": """Transform the following text to sound more friendly and approachable:
- Use warm, welcoming language
- Add encouraging phrases where appropriate
- Make it sound like a helpful friend explaining something
- Include empathetic expressions
- Keep the information accurate and helpful""",

"storytelling": """Rewrite the following text in a storytelling style:
- Use narrative elements where appropriate
- Add descriptive language and examples
- Make it engaging and relatable
- Include personal touches and anecdotes if suitable
- Maintain the factual content""",

"layman_terms": """Rewrite the following text in simple, layman-friendly language:
- Avoid technical terms and jargon
- Use everyday examples or analogies
- Break down complex ideas into plain language
- Use short, clear sentences
- Make it easy for someone with no background knowledge to understand""",

"indian_english": """Rewrite the following text in Indian English:
- Use sentence structures common in Indian usage
- Include polite and respectful expressions (e.g., 'kindly', 'please do the needful') where suitable
- Reflect a slightly formal but approachable tone
- Use Indian expressions or culturally familiar terms where appropriate
- Keep the original meaning intact while sounding locally natural"""

        }
        base_prompt = style_prompts.get(style, style_prompts["conversational"])
        return f"""{base_prompt}

Original text:
{text}

Humanized version:"""
    
    def humanize_text(self, text: str, style: str = "conversational", temperature: float = 0.7) -> str:
        """Humanize the input text using ChatGroq"""
        try:
            if self.temperature != temperature:
                self.update_model_settings(self.model, temperature)
            
            prompt = self.create_humanization_prompt(text, style)
            
            messages = [
              SystemMessage(content="""
                            You are a highly skilled writing assistant specialized in transforming robotic or overly technical text into clear, engaging, and human-like blog-style content.

                            Your task is to:
                            - Maintain the original meaning and factual accuracy.
                            - Use a natural, conversational tone appropriate for blog readers.
                            - Organize content using clear **headings**, **subheadings**, and **bullet points** or **numbered lists** where appropriate.
                            - Highlight key ideas with formatting if needed (e.g., bold or italics).
                            - Add helpful transitions and context to improve readability.
                            - Ensure the final result reads like a professional and informative blog post.
                            - keep the Blog Post atleast 1500 words Minimum
                                """
                                ),

                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            cleaned_response = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
            return cleaned_response
        
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    
    with st.sidebar:
        st.header("Configuration")
        api_key = "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"
        style = st.selectbox("Humanization Style", ["conversational", "professional", "friendly", "storytelling","layman_terms","indian_english"])
        temperature = st.slider("Creativity Level", 0.1, 1.0, 0.7, 0.1)
        model_options = [
            "deepseek-r1-distill-llama-70b",
            "llama3-8b-8192", 
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        selected_model = st.selectbox("Model", model_options)
    
    try:
        rag_system = TextHumanizerRAG(api_key)
        rag_system.update_model_settings(selected_model, temperature)
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Text")
        original_text = st.text_area("Enter your text here:", height=300)
        word_count_original = len(original_text.split())
        st.markdown(f"**Word count:** {word_count_original}")
        if st.button("Humanize Text", type="primary"):
            if original_text.strip():
                with st.spinner("Humanizing text..."):
                    humanized = rag_system.humanize_text(original_text, style, temperature)
                    st.session_state.humanized_text = humanized
            else:
                st.warning("Please enter some text to humanize")

    with col2:
        st.header("Humanized Text")
        if hasattr(st.session_state, 'humanized_text'):
            # Create a container for the text area and copy icon
            text_container = st.container()
            with text_container:
                st.text_area("Humanized Result:", value=st.session_state.humanized_text, height=300, key="humanized_output")
                word_count_humanized = len(st.session_state.humanized_text.split())
                st.markdown(f"**Word count:** {word_count_humanized}")
                
                # Add copy icon with JavaScript
                copy_html = f"""
                <div style="position: relative; margin-top: -40px; margin-bottom: 20px; text-align: right; margin-right: 10px;">
                    <button onclick="copyToClipboard()" style="
                        background: transparent;
                        border: none;
                        cursor: pointer;
                        font-size: 18px;
                        color: #666;
                        padding: 5px;
                        border-radius: 3px;
                        transition: color 0.2s;
                    " 
                    onmouseover="this.style.color='#333'"
                    onmouseout="this.style.color='#666'"
                    title="Copy to clipboard">
                        📋
                    </button>
                </div>
                
                <script>
                function copyToClipboard() {{
                    const text = `{st.session_state.humanized_text.replace('`', '\\`').replace('${', '\\${').replace(chr(10), '\\n').replace(chr(13), '\\r')}`;
                    navigator.clipboard.writeText(text).then(function() {{
                        // Show success message
                        const button = event.target;
                        const originalText = button.innerHTML;
                        button.innerHTML = '✅';
                        button.style.color = '#28a745';
                        setTimeout(function() {{
                            button.innerHTML = originalText;
                            button.style.color = '#666';
                        }}, 1500);
                    }}).catch(function(err) {{
                        alert("Failed to copy text. Please copy manually.");
                    }});
                }}
                </script>
                """
                st.components.v1.html(copy_html, height=50)

if __name__ == "__main__":
    main()