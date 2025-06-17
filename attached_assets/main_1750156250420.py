from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import json
import time
import hashlib
import os
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from supabase import create_client
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import openai
import google.generativeai as genai
from typing import Dict, Optional
from datetime import datetime
import io
import asyncio
import re

os.makedirs("templates", exist_ok=True)

# Pydantic Models for Request Validation
class ProcessDataRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Text content to process")
    user_id: str = Field(..., min_length=1, description="User identifier")
    session_id: str = Field(..., min_length=1, description="Session identifier")
    chatbot_id: Optional[str] = Field(None, description="Chatbot identifier for filtering documents")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Additional metadata")

class SimilarityScore(BaseModel):
    id: str
    similarity: float
    metadata: Dict
    scores: Optional[Dict] = None

class DocumentResponse(BaseModel):
    id: str
    first_5_embedding: List[float]
    embedding_length: int
    metadata: Dict
    similarity: float

class ProcessDataResponse(BaseModel):
    status: str
    most_similar_content: Optional[Dict]
    chat_response: Optional[str]
    query_embedding_length: int
    all_documents: List[DocumentResponse]
    similarities: List[SimilarityScore]
    timing: Dict[str, float]


class WebCrawler:

    def __init__(self):
        self.headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept':
            'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        self.session = requests.Session()

    def is_valid_url(self, base_url, url):
        try:
            base_domain = urlparse(base_url).netloc
            url_domain = urlparse(url).netloc
            invalid_extensions = ('.css', '.js', '.png', '.jpg', '.jpeg',
                                  '.gif', '.pdf', '.doc', '.docx', '.svg',
                                  '.ico', '.woff', '.woff2', '.ttf', '.eot')
            return (base_domain == url_domain
                    and not url.endswith(invalid_extensions) and '#' not in url
                    and 'mailto:' not in url and 'tel:' not in url)
        except:
            return False

    def check_sitemap(self, base_url):
        try:
            sitemap_url = urljoin(base_url, '/sitemap.xml')
            response = self.session.get(sitemap_url,
                                        headers=self.headers,
                                        timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                urls = []
                for url in root.findall(
                        './/{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    urls.append(url.text)
                return urls
            return []
        except:
            return []

    def fetch_links(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = self.session.get(url, headers=self.headers, timeout=10)
            if response.status_code == 429:
                time.sleep(30)
                response = self.session.get(url,
                                            headers=self.headers,
                                            timeout=10)

            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            seen_urls = set()

            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href')
                full_url = urljoin(url, href)
                if self.is_valid_url(url,
                                     full_url) and full_url not in seen_urls:
                    seen_urls.add(full_url)
                    link_text = a_tag.get_text(strip=True) or full_url
                    links.append({
                        'url':
                        full_url,
                        'text':
                        link_text[:100] +
                        '...' if len(link_text) > 100 else link_text
                    })

            sitemap_urls = self.check_sitemap(url)
            for sitemap_url in sitemap_urls:
                if sitemap_url not in seen_urls:
                    seen_urls.add(sitemap_url)
                    links.append({'url': sitemap_url, 'text': 'From Sitemap'})

            return {
                'status': 'success',
                'message': f'Found {len(links)} internal links',
                'links': links
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def crawl_pages(self, urls):
        data = {'pages': {}, 'metadata': {'start_time': time.time()}}
        total = len(urls)

        for index, url in enumerate(urls, 1):
            try:
                response = self.session.get(url,
                                            headers=self.headers,
                                            timeout=10)
                if response.status_code == 429:
                    time.sleep(30)
                    response = self.session.get(url,
                                                headers=self.headers,
                                                timeout=10)

                soup = BeautifulSoup(response.text, 'html.parser')
                tables = []
                for table in soup.find_all('table'):
                    tables.append(str(table))

                data['pages'][url] = {
                    'title':
                    soup.title.string if soup.title else '',
                    'meta_description':
                    soup.find('meta', {'name': 'description'})['content']
                    if soup.find('meta', {'name': 'description'}) else '',
                    'html_tables':
                    tables,
                    'text_content':
                    soup.get_text(strip=True),
                    'status':
                    'success',
                    'timestamp':
                    time.time()
                }

                yield {
                    'status': 'crawling',
                    'progress': (index / total) * 100,
                    'current_url': url
                }

                time.sleep(2)

            except Exception as e:
                data['pages'][url] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }

        filename = f"crawl_{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        yield {'status': 'complete', 'download_url': f'/download/{filename}'}


app = FastAPI(title="Document Processing API")
templates = Jinja2Templates(directory="templates")

# Mount static files to serve audio
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/notification-sound")
async def get_notification_sound():
    """Serve the notification sound file"""
    import os
    if os.path.exists("notification.mp3"):
        return FileResponse(
            "notification.mp3", 
            media_type="audio/mpeg",
            filename="notification.mp3",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Type": "audio/mpeg",
                "Cache-Control": "public, max-age=3600"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")


@app.post("/generate_greeting/")
async def generate_greeting(chatbot_id: str):
    """Generate greeting message for a chatbot based on its documents"""
    try:
        # First check if greeting already exists and is not NULL
        existing_greeting = supabase.table('chatbot_themes').select('*').eq(
            'chatbot_id', chatbot_id).execute()

        if existing_greeting.data and existing_greeting.data[0].get(
                'greeting'):
            return {
                "status": "success",
                "chatbot_id": chatbot_id,
                "greeting": existing_greeting.data[0]['greeting'],
                "message": "Existing greeting returned"
            }

        # If no greeting exists or is NULL, fetch documents to generate one
        result = supabase.table('documents').select('*').eq(
            'chatbot_id', chatbot_id).execute()

        if not result.data:
            raise HTTPException(status_code=404,
                                detail="No documents found for this chatbot")

        # Get the first document's content
        doc_content = result.data[0]['content']

        # Generate greeting using OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""Based on the following content about an organization, create a warm, professional greeting message that a chatbot would use when starting a conversation with a user. The greeting should:

1. Be welcoming and professional
2. Briefly mention the organization's core purpose
3. Invite the user to ask questions
4. Not use any emojis
5. Be concise (max 3-4 sentences)

Content about the organization:
{doc_content[:2000]}  # Using first 2000 chars for context

Generate only the greeting message,  Do remember to ask about name,email and phone number to the customer in the greeting at the end."""

        completion = client.chat.completions.create(model="gpt-4",
                                                    messages=[{
                                                        "role": "user",
                                                        "content": prompt
                                                    }])

        greeting = completion.choices[0].message.content.strip()

        # Update existing row with new greeting
        if existing_greeting.data:
            # Update existing row where greeting is NULL
            update_result = supabase.table('chatbot_themes').update({
                'greeting':
                greeting
            }).eq('chatbot_id', chatbot_id).is_('greeting', 'null').execute()
        else:
            # Insert new row if no row exists for this chatbot_id
            update_result = supabase.table('chatbot_themes').insert({
                'chatbot_id':
                chatbot_id,
                'greeting':
                greeting
            }).execute()

        return {
            "status": "success",
            "chatbot_id": chatbot_id,
            "greeting": greeting
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to generate greeting: {str(e)}")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/notification-test", response_class=HTMLResponse)
async def notification_test(request: Request):
    return templates.TemplateResponse("notification_test.html", {"request": request})

@app.get("/check-audio")
async def check_audio():
    """Check if audio file exists and get its properties"""
    import os
    if os.path.exists("notification.mp3"):
        file_size = os.path.getsize("notification.mp3")
        return {
            "status": "found",
            "file_size": file_size,
            "file_path": "notification.mp3",
            "message": "Audio file is accessible"
        }
    else:
        return {
            "status": "not_found",
            "message": "notification.mp3 file not found",
            "suggestion": "Please upload a valid MP3 file named 'notification.mp3'"
        }


@app.get("/fetch-links/")
async def fetch_links(url: str):
    try:
        crawler = WebCrawler()
        result = crawler.fetch_links(url)
        if result['status'] == 'success':
            return result
        else:
            raise HTTPException(status_code=500, detail=result['message'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from typing import List


@app.post("/crawl/")
async def crawl(urls: List[str]):
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    crawler = WebCrawler()

    async def event_stream():
        async for data in crawler.crawl_pages(urls):
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/download/{filename}")
async def download_file(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


SUPABASE_URL = "https://zsivtypgrrcttzhtfjsf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpzaXZ0eXBncnJjdHR6aHRmanNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzgzMzU5NTUsImV4cCI6MjA1MzkxMTk1NX0.3cAMZ4LPTqgIc8z6D8LRkbZvEhP_ffI3Wka0-QDSIys"
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyCIc2_BOotAbuRDwsJ7fUu1EdQ7sGFi_So"
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)


async def extract_text_from_pdf(file: bytes) -> str:
    """Extract text from PDF file"""
    return extract_pdf_text(io.BytesIO(file))


async def extract_text_from_docx(file: bytes) -> str:
    """Extract text from DOCX file"""
    doc = Document(io.BytesIO(file))
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


async def scrape_text_from_url(url: str) -> str:
    """Scrape text content from URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return " ".join(soup.stripped_strings)


async def generate_embedding(text: str) -> list:
    """Generate embedding using OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # First clean and standardize the text
        text = text.strip()
        # Then truncate to OpenAI's token limit (approximately 8191 characters)
        text = text[:8191]

        response = client.embeddings.create(input=text,
                                            model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI embedding generation failed: {str(e)}")


@app.post("/upload-file/")
async def upload_file(file: UploadFile):
    """Handle file upload and text extraction"""
    content = await file.read()

    if file.filename.lower().endswith('.pdf'):
        text = await extract_text_from_pdf(content)
    elif file.filename.lower().endswith('.docx'):
        text = await extract_text_from_docx(content)
    elif file.filename.lower().endswith('.json'):
        try:
            json_content = json.loads(content.decode('utf-8'))
            # Convert JSON to string representation
            text = json.dumps(json_content, indent=2)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    return {"text": text}


@app.post("/scrape-url/")
async def scrape_url(url: str):
    """Handle URL scraping"""
    try:
        text = await scrape_text_from_url(url)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Failed to scrape URL: {str(e)}")


@app.post("/crawl_url")
async def crawl_url(urls: list):
    """Handle multiple URL crawling"""
    try:
        results = {}
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = " ".join(soup.stripped_strings)
                results[url] = {
                    "status": "success",
                    "text": text,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                results[url] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        return results
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Failed to crawl URLs: {str(e)}")


@app.get("/all-documents/")
async def get_all_documents():
    """Fetch all documents from Supabase"""
    try:
        result = supabase.table('documents').select('*').execute()
        return {"documents": result.data}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to fetch documents: {str(e)}")


@app.post("/chat_response/")
async def chat_response(request: Dict):
    query = request.get("query")
    context = request.get("context")
    if not query or not context:
        raise HTTPException(status_code=422,
                            detail="Both query and context are required")
    """Generate chat response using OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"Given the following context, please answer the query:\n\nContext: {context}\n\nQuery: {query}"

        completion = client.chat.completions.create(model="gpt-4",
                                                    messages=[{
                                                        "role": "user",
                                                        "content": prompt
                                                    }])

        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat response generation failed: {str(e)}")


# Helper functions for process_data decomposition
async def _generate_embedding_helper(content: str) -> list:
    """Generate embedding for given content"""
    return await generate_embedding(content)

async def _retrieve_relevant_documents(chatbot_id: str, embedding: list) -> tuple[list, list]:
    """Retrieve and rank documents by similarity"""
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Fetch documents filtered by chatbot_id
    query = supabase.table('documents').select('*')
    if chatbot_id:
        query = query.eq('chatbot_id', chatbot_id)
    all_docs = query.execute()
    
    # Calculate similarities with all documents using all embedding types
    similarities = []
    
    for doc in all_docs.data:
        # Skip documents that are queries
        if doc.get("metadata", {}).get("type") == "query":
            continue
            
        # Get content embedding
        content_embedding = doc["embedding"] if isinstance(
            doc["embedding"], list) else json.loads(doc["embedding"])
        content_score = cosine_similarity(embedding, content_embedding)
        
        # Get source embedding
        source_embedding = doc["source_embedding"] if isinstance(
            doc["source_embedding"], list) else json.loads(
                doc["source_embedding"])
        source_score = cosine_similarity(embedding, source_embedding)
        
        # Get keyword embedding
        keyword_embedding = doc["keyword_embedding"] if isinstance(
            doc["keyword_embedding"], list) else json.loads(
                doc["keyword_embedding"])
        keyword_score = cosine_similarity(embedding, keyword_embedding)
        
        # Calculate final weighted score
        final_score = (0.6 * content_score) + (0.2 * source_score) + (
            0.2 * keyword_score)
        
        similarities.append({
            "id": doc["id"],
            "similarity": float(final_score),
            "metadata": doc["metadata"],
            "scores": {
                "content": float(content_score),
                "source": float(source_score),
                "keyword": float(keyword_score),
                "final": float(final_score)
            }
        })
    
    # Sort by final similarity score
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return all_docs.data, similarities

async def _manage_conversation_history(user_id: str, session_id: str) -> str:
    """Retrieve and format conversation history"""
    # Get previous conversation history
    conversation_history = supabase.table('chat_messages')\
        .select('*')\
        .eq('user_id', user_id)\
        .eq('session_id', session_id)\
        .order('timestamp', desc=True)\
        .limit(6)\
        .execute()  # Get last 6 messages (3 pairs of user-assistant interactions)
    
    # Format conversation history in chronological order
    conversation_messages = []
    if conversation_history.data:
        # Reverse to get chronological order and format messages
        for msg in reversed(conversation_history.data):
            role = "User" if msg['message_type'] == 'user' else "Assistant"
            conversation_messages.append(f"{role}: {msg['message']}")
    
    # Convert to string with newlines between messages
    return "\n".join(conversation_messages) if conversation_messages else ""

async def _generate_llm_prompt(content: str, conversation_context: str, top_similar_docs: list, 
                              chatbot_id: str, additional_instruction: str = "") -> str:
    """Generate comprehensive LLM prompt"""
    # Fetch theme settings from chatbot_themes
    response_style = ""
    greeting_message = ""
    if chatbot_id:
        theme_result = supabase.table('chatbot_themes').select(
            'instruction,response_tone,response_length,greeting'
        ).eq('chatbot_id', chatbot_id).execute()

        if theme_result.data and theme_result.data[0]:
            # Get instruction if present
            if theme_result.data[0].get('instruction'):
                additional_instruction = f"\nAdditional Instructions: {theme_result.data[0]['instruction']}\n"

            if theme_result.data[0].get('greeting'):
                greeting_message = f"{theme_result.data[0]['greeting']}"

            # Add tone and length settings if present
            tone = theme_result.data[0].get('response_tone')
            length = theme_result.data[0].get('response_length')

            if tone or length:
                response_style = "\nResponse Style Requirements:"
                if tone:
                    response_style += f"\n- Maintain a {tone} tone throughout the response"
                if length:
                    word_limit = "50-60" if length == "concise" else "100-110"
                    response_style += f"\n- Keep response within {word_limit} words"

    # Check if the query is a greeting
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon',
        'good evening', 'greetings'
    ]
    is_greeting = content.lower().strip() in greetings

    # Build response style requirements
    style_requirements = []
    if tone := theme_result.data[0].get('response_tone'):
        style_requirements.append(f"maintain a {tone} tone throughout")
    if length := theme_result.data[0].get('response_length'):
        word_limit = "50-60" if length == "concise" else "100-110"
        style_requirements.append(f"keep response within {word_limit} words")

    style_instructions = "\n- ".join(style_requirements) if style_requirements else ""

    # Build context from top 3 documents
    context_sections = []
    for i, doc in enumerate(top_similar_docs[:3], 1):
        # Get document source name from metadata
        metadata = doc.get('metadata', {})
        if metadata.get('filename'):
            source_name = metadata['filename']
        elif metadata.get('url'):
            source_name = metadata['url']
        elif metadata.get('title'):
            source_name = metadata['title']
        else:
            source_name = f"Document {i}"

        context_sections.append(f"Source: {source_name}\n{doc['content']}")

    full_context = "\n\n".join(context_sections)

    prompt = f"""You are an AI assistant representing an organization.{additional_instruction}

Previous Conversation:
{conversation_context}

Response Requirements:
- {style_instructions}
- Do not use emojis or emoticons
- Connect responses to previous conversation context when relevant
- Maintain information consistency across queries
- Structure response in clear, focused points
- First look at the document source names below and identify which document(s) are most relevant to answer the user query

CRITICAL FORMATTING REQUIREMENTS:
- Use **text** for bolding only to emphasize key terms, names, critical numbers, or short phrases that are essential for the user to immediately notice. Do not bold entire sentences or paragraphs.
- Ensure bolded text is always correctly enclosed within two asterisks (**) on both sides, with no spaces between the asterisks and the text (e.g., **Example**).
- Bolding should enhance readability and highlight crucial information, not disrupt the flow. Use it sparingly for maximum impact.
- Only bold information that is directly and unequivocally stated in the provided context documents.
- Use ONLY hyphens (-) followed by a space for bullet points. Do NOT use asterisks (*), dots (•), or any other characters for list items.
- Do NOT use markdown headings (#, ##, ###). Write section titles as natural language sentences.
- Follow paragraph-then-bullets structure: Start with a concise introductory paragraph summarizing the answer, followed by detailed information presented in clear, distinct bullet points using hyphens (-).

                {"IF THIS IS A GREETING AND NO PREVIOUS CONVERSATION:" if is_greeting and not conversation_context else ""}
                1. {"Start with a warm welcome using first-person plural pronouns (we, our, us). Don't give much detail of organization." if is_greeting and not conversation_context else "Maintain a natural conversation flow, as this is a continuing discussion."}
                   {"Focus on giving possible questions users might ask, keeping them relevant to educational institute enquiries only." if is_greeting and not conversation_context else ""}

                2. {"If this is not a greeting, provide information following paragraph-then-bullets structure:" if not is_greeting else "Suggest possible areas of interest following paragraph-then-bullets structure:"}
                   - Start with a concise introductory paragraph summarizing your answer
                   - Follow with bullet points using hyphens (-) and spaces for detailed information
                   - Use clear, concise points with relevant details from the context
                   - Maintain a professional tone without emojis
                   - Keep each point focused and specific
                   - Bold only key terms, names, or critical numbers using **text** format

                3. DYNAMIC FOLLOW-UP QUESTION GENERATION:
                   - Generate follow-up questions that are highly specific and directly related to the content of your response
                   - Create "next logical step" inquiries based on what you just discussed
                   - Each question must be exactly 4-5 words
                   - Questions must be actionable, specific, and distinct from each other
                   - Questions should encourage deeper exploration of the topic you just covered
                   - Do NOT use any formatting (no hashtags, bold, asterisks, or styling) in the follow-up questions
                   - Follow this EXACT format:
                   
                   "Would you like to ask any further questions?"
                   1. [First follow-up question (4-5 words)]
                   2. [Second follow-up question (4-5 words)]

                Context Documents (choose the most relevant one(s) to answer the query):
                {full_context}

                Please answer this query: {content}

                Remember to:
- Keep the tone professional and engaging without using any emojis
- {"Focus on introducing the organization and suggesting relevant questions" if is_greeting else "Provide specific, relevant information"}
- Follow the paragraph-then-bullets structure strictly
- Use hyphens (-) for bullet points, never asterisks (*) or dots (•)
- Write section titles as natural sentences, not markdown headings
- Bold only essential information using **text** format
- Generate contextual follow-up questions directly related to your response content
- Ensure follow-up questions are 4-5 words each with no formatting or styling
- Make follow-up questions actionable and encourage deeper topic exploration"""

    return prompt

async def _generate_chatbot_response(prompt: str, user_id: str, session_id: str, 
                                   greeting_message: str = "") -> tuple[str, int, int]:
    """Generate chatbot response using OpenAI"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Count tokens (words) in prompt
    prompt_tokens = len(prompt.split())
    print(f"\nPrompt Token Count: {prompt_tokens} tokens (words)")
    
    now = datetime.utcnow()
    messages = []
    current_input_tokens = prompt_tokens
    
    # Check if conversation exists first and handle 5-minute timeout
    existing_convo = supabase.table('testing_zaps2')\
        .select('*')\
        .eq('chatbot_id', user_id)\
        .eq('session_id', session_id)\
        .execute()
    
    should_create_new = False
    if existing_convo.data:
        # Check if conversation is older than 5 minutes
        try:
            from datetime import datetime, timedelta
            created_at_str = existing_convo.data[0].get('created_at')
            if created_at_str:
                created_at = datetime.fromisoformat(
                    created_at_str.replace('Z', '+00:00'))
                time_diff = datetime.now() - created_at.replace(tzinfo=None)
                if time_diff > timedelta(minutes=5):
                    # Modify existing session_id by appending timestamp
                    current_timestamp = int(time.time())
                    new_session_id = f"{session_id}_{current_timestamp}"
                    
                    # Update existing conversation's session_id
                    supabase.table('testing_zaps2').update({
                        'session_id': new_session_id
                    }).eq('chatbot_id', user_id).eq('session_id', session_id).execute()
                    
                    should_create_new = True
        except Exception as e:
            # If there's any error parsing time, create new conversation
            should_create_new = True
    
    # Only add greeting if creating new conversation and greeting exists
    if (not existing_convo.data or should_create_new) and greeting_message:
        messages.append({
            "role": "assistant",
            "content": greeting_message
        })
    
    messages.append({"role": "user", "content": prompt})
    
    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}])
    chat_response = completion.choices[0].message.content
    
    # Count tokens in response
    response_tokens = len(chat_response.split())
    print(f"\nResponse Token Count: {response_tokens} tokens (words)")
    
    messages.append({"role": "assistant", "content": chat_response})
    
    if existing_convo.data and not should_create_new:
        # Update existing conversation
        existing_row = existing_convo.data[0]
        existing_messages = existing_row.get('messages', [])
        updated_messages = existing_messages + messages
        
        # Update token counts
        input_tokens = (existing_row.get('input_tokens') or 0) + current_input_tokens
        output_tokens = (existing_row.get('output_tokens') or 0) + response_tokens
        
        supabase.table('testing_zaps2').update({
            'messages': updated_messages,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'date_of_convo': now.date().isoformat(),
            'time_of_convo': now.time().isoformat()
        }).eq('chatbot_id', user_id).eq('session_id', session_id).execute()
    else:
        # Create new conversation
        supabase.table('testing_zaps2').insert({
            'chatbot_id': user_id,
            'session_id': session_id,
            'messages': messages,
            'input_tokens': current_input_tokens,
            'output_tokens': response_tokens,
            'date_of_convo': now.date().isoformat(),
            'time_of_convo': now.time().isoformat(),
            'created_at': now.isoformat()
        }).execute()
    
    return chat_response, current_input_tokens, response_tokens

@app.post("/process-data")
async def process_data(request: Dict):
    # Validate required parameters
    content = request.get("content")
    user_id = request.get("user_id")
    session_id = request.get("session_id")
    if not all([content, user_id, session_id]):
        raise HTTPException(
            status_code=422,
            detail=
            "Missing required parameters: content, user_id, and session_id are mandatory"
        )

    metadata = request.get("metadata", {})
    chatbot_id = request.get("chatbot_id")
    timing_info = {}  #Added to store timing information

    """Process text and store in Supabase with content and embedding"""
    try:
        # Step 1: Generate embedding
        embedding_start = time.time()
        embedding = await _generate_embedding_helper(content)
        timing_info['embedding_generation'] = time.time() - embedding_start

        # Step 2: Retrieve and rank relevant documents
        document_search_start = time.time()
        all_docs, similarities = await _retrieve_relevant_documents(chatbot_id, embedding)
        timing_info['fetch_all_docs'] = time.time() - document_search_start
        timing_info['cosine_similarity_process'] = 0.0  # Included in retrieve function

        # Step 3: Handle fallback scenarios and get top documents
        top_similar_docs = []
        chat_response = None
        most_similar_doc = None

        if similarities and similarities[0]["similarity"] < 0.35:
            # Get fallback message from chatbot_themes
            if chatbot_id:
                fallback_result = supabase.table('chatbot_themes').select(
                    'fallback_message').eq('chatbot_id', chatbot_id).execute()
                if fallback_result.data and fallback_result.data[0] and fallback_result.data[0].get('fallback_message'):
                    chat_response = fallback_result.data[0]['fallback_message']
                else:
                    chat_response = "Oops! I am not able to get context to answer your query!"
        else:
            if similarities:  # Use the top 3 most similar documents
                # Get top 3 documents
                for i in range(min(3, len(similarities))):
                    doc_id = similarities[i]["id"]
                    doc = next((doc for doc in all_docs if doc["id"] == doc_id), None)
                    if doc:
                        top_similar_docs.append(doc)

                # Keep the first document for backward compatibility
                most_similar_doc = top_similar_docs[0] if top_similar_docs else None

        # Step 4: Generate chat response if no fallback
        chat_response_start = time.time()
        if not chat_response and most_similar_doc and content:
            try:
                # Step 4a: Manage conversation history
                conversation_context = await _manage_conversation_history(user_id, session_id)
                
                # Step 4b: Generate LLM prompt
                prompt = await _generate_llm_prompt(content, conversation_context, top_similar_docs, chatbot_id)
                
                # Print the final prompt for debugging
                print("\nFinal Prompt Sent to OpenAI:")
                print("-" * 80)
                print(prompt)
                print("-" * 80)
                
                # Step 4c: Generate chatbot response
                chat_response, input_tokens, output_tokens = await _generate_chatbot_response(
                    prompt, user_id, session_id)
                
                # Apply comprehensive text formatting and validation
                chat_response = format_chatbot_response(chat_response)
                
            except Exception as e:
                print(f"Chat response generation failed: {str(e)}")
                
        timing_info['chat_response_generation'] = time.time() - chat_response_start

        response_start = time.time()
        response_data = {
            "status":
            "success",
            "most_similar_content": {
                "content":
                most_similar_doc["content"] if most_similar_doc else None,
                "similarity":
                similarities[1]["similarity"]
                if len(similarities) > 1 else None,
                "metadata":
                most_similar_doc["metadata"] if most_similar_doc else None
            },
            "chat_response":
            chat_response,
            "query_embedding_length":
            len(embedding),
            "all_documents": [{
                "id":
                doc["id"],
                "first_5_embedding":
                doc["embedding"][:5] if isinstance(doc["embedding"], list) else
                json.loads(doc["embedding"])[:5],
                "embedding_length":
                len(doc["embedding"]) if isinstance(doc["embedding"], list)
                else len(json.loads(doc["embedding"])),
                "metadata":
                doc["metadata"],
                "similarity":
                next((s["similarity"]
                      for s in similarities if s["id"] == doc["id"]), 0)
            } for doc in all_docs.data],
            "similarities":
            similarities,
            "timing":
            timing_info  # Added timing information to response
        }
        timing_info['response_preparation'] = time.time() - response_start

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Processing failed: {str(e)}")


##################################
@app.post("/process-data/whatsapp/old")
async def process_data_whastapp_old(request: Dict):
    # Validate required parameters
    content = request.get("content")
    user_id = request.get("user_id")
    session_id = request.get("session_id")
    if not all([content, user_id, session_id]):
        raise HTTPException(
            status_code=422,
            detail=
            "Missing required parameters: content, user_id, and session_id are mandatory"
        )

    metadata = request.get("metadata", {})
    chatbot_id = request.get("chatbot_id")
    timing_info = {}  #Added to store timing information

    # Log user message first

    # supabase.table('chat_messages').insert(user_message_data).execute()
    """Process text and store in Supabase with content and embedding"""
    try:
        # Generate embedding using OpenAI
        embedding_start = time.time()
        embedding = await generate_embedding(content)
        timing_info['embedding_generation'] = time.time() - embedding_start

        document_search_start = time.time()

        # Fetch documents filtered by chatbot_id
        query = supabase.table('documents').select('*')
        if chatbot_id:
            query = query.eq('chatbot_id', chatbot_id)
        all_docs = query.execute()
        timing_info['fetch_all_docs'] = time.time() - document_search_start

        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Calculate similarities with all documents using all embedding types
        similarities = []
        cosine_similarity_start = time.time()

        for doc in all_docs.data:
            # Skip documents that are queries
            if doc.get("metadata", {}).get("type") == "query":
                continue

            # Get content embedding
            content_embedding = doc["embedding"] if isinstance(
                doc["embedding"], list) else json.loads(doc["embedding"])
            content_score = cosine_similarity(embedding, content_embedding)

            # Get source embedding
            source_embedding = doc["source_embedding"] if isinstance(
                doc["source_embedding"], list) else json.loads(
                    doc["source_embedding"])
            source_score = cosine_similarity(embedding, source_embedding)

            # Get keyword embedding
            keyword_embedding = doc["keyword_embedding"] if isinstance(
                doc["keyword_embedding"], list) else json.loads(
                    doc["keyword_embedding"])
            keyword_score = cosine_similarity(embedding, keyword_embedding)

            # Calculate final weighted score
            final_score = (0.6 * content_score) + (0.2 * source_score) + (
                0.2 * keyword_score)

            similarities.append({
                "id": doc["id"],
                "similarity": float(final_score),
                "metadata": doc["metadata"],
                "scores": {
                    "content": float(content_score),
                    "source": float(source_score),
                    "keyword": float(keyword_score),
                    "final": float(final_score)
                }
            })

        # Sort by final similarity score
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Get content of top 3 most similar documents
        top_similar_docs = []
        chat_response = None

        if similarities and similarities[0]["similarity"] < 0.35:
            # Get fallback message from chatbot_themes
            if chatbot_id:
                fallback_result = supabase.table('chatbot_themes').select(
                    'fallback_message').eq('chatbot_id', chatbot_id).execute()
                if fallback_result.data and fallback_result.data[
                        0] and fallback_result.data[0].get('fallback_message'):
                    chat_response = fallback_result.data[0]['fallback_message']
                else:
                    chat_response = "Oops! I am not able to get context to answer your query!"
        else:
            if similarities:  # Use the top 3 most similar documents
                # Get top 3 documents
                for i in range(min(3, len(similarities))):
                    doc_id = similarities[i]["id"]
                    doc = next(
                        (doc for doc in all_docs.data if doc["id"] == doc_id),
                        None)
                    if doc:
                        top_similar_docs.append(doc)

                # Keep the first document for backward compatibility
                most_similar_doc = top_similar_docs[
                    0] if top_similar_docs else None

        timing_info['cosine_similarity_process'] = time.time(
        ) - cosine_similarity_start
        chat_response_start = time.time()

        # Only proceed with OpenAI response if similarity >= 0.65 and no fallback message set
        if not chat_response:
            # Generate chat response if similar document is found

            # Get previous conversation history
            conversation_history = supabase.table('chat_messages')\
                .select('*')\
                .eq('user_id', user_id)\
                .eq('session_id', session_id)\
                .order('timestamp', desc=True)\
                .limit(6)\
                .execute()  # Get last 6 messages (3 pairs of user-assistant interactions)

            # Format conversation history in chronological order
            conversation_messages = []
            if conversation_history.data:
                # Reverse to get chronological order and format messages
                for msg in reversed(conversation_history.data):
                    role = "User" if msg[
                        'message_type'] == 'user' else "Assistant"
                    conversation_messages.append(f"{role}: {msg['message']}")

            # Convert to string with newlines between messages
            conversation_context = "\n".join(
                conversation_messages) if conversation_messages else ""

            if most_similar_doc and content:  # content here is the query
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    # Fetch theme settings from chatbot_themes
                    additional_instruction = ""
                    response_style = ""
                    greeting_message = ""
                    if chatbot_id:
                        theme_result = supabase.table('chatbot_themes').select(
                            'instruction,response_tone,response_length,greeting'
                        ).eq('chatbot_id', chatbot_id).execute()

                        if theme_result.data and theme_result.data[0]:
                            # Get instruction if present
                            if theme_result.data[0].get('instruction'):
                                additional_instruction = f"\nAdditional Instructions: {theme_result.data[0]['instruction']}\n"

                            if theme_result.data[0].get('greeting'):
                                greeting_message = f"{theme_result.data[0]['greeting']}"

                            # Add tone and length settings if present
                            tone = theme_result.data[0].get('response_tone')
                            length = theme_result.data[0].get(
                                'response_length')

                            if tone or length:
                                response_style = "\nResponse Style Requirements:"
                                if tone:
                                    response_style += f"\n- Maintain a {tone} tone throughout the response"
                                if length:
                                    word_limit = "50-60" if length == "concise" else "100-110"
                                    response_style += f"\n- Keep response within {word_limit} words"

                    # Check if the query is a greeting
                    greetings = [
                        'hi', 'hello', 'hey', 'good morning', 'good afternoon',
                        'good evening', 'greetings'
                    ]
                    is_greeting = content.lower().strip() in greetings

                    # Build response style requirements
                    style_requirements = []
                    if tone := theme_result.data[0].get('response_tone'):
                        style_requirements.append(
                            f"maintain a {tone} tone throughout")
                    if length := theme_result.data[0].get('response_length'):
                        word_limit = "50-60" if length == "concise" else "100-110"
                        style_requirements.append(
                            f"keep response within {word_limit} words")

                    style_instructions = "\n- ".join(
                        style_requirements) if style_requirements else ""

                    # Build context from top 3 documents
                    context_sections = []
                    for i, doc in enumerate(top_similar_docs[:3], 1):
                        # Get document source name from metadata
                        metadata = doc.get('metadata', {})
                        if metadata.get('filename'):
                            source_name = metadata['filename']
                        elif metadata.get('url'):
                            source_name = metadata['url']
                        elif metadata.get('title'):
                            source_name = metadata['title']
                        else:
                            source_name = f"Document {i}"

                        context_sections.append(
                            f"Source: {source_name}\n{doc['content']}")

                    full_context = "\n\n".join(context_sections)

                    prompt = f"""You are an AI assistant representing an organization.{additional_instruction}

Previous Conversation:
{conversation_context}

Response Requirements:
- {style_instructions}
- Do not use emojis or emoticons
- Connect responses to previous conversation context when relevant
- Maintain information consistency across queries
- Structure response in clear, focused points
-Please use markdown format to make new lines in teh response, such as use %0D%0A for a line break and nothing else ok, as i want to display these responses in markdown format in whatsapp.
- First look at the document source names below and identify which document(s) are most relevant to answer the user query

                    {"IF THIS IS A GREETING AND NO PREVIOUS CONVERSATION:" if is_greeting and not conversation_messages else ""}
                    1. {"Start with a warm welcome using first-person plural pronouns (we, our, us). Don't give much detail of organization." if is_greeting and not conversation_messages else "Maintain a natural conversation flow, as this is a continuing discussion."}
                       {"Focus on giving possible questions users might ask, keeping them relevant to educational institute enquiries only." if is_greeting and not conversation_messages else ""}

                    2. {"If this is not a greeting, provide information as bullet points:" if not is_greeting else "Suggest possible areas of interest:"}
                       • Use clear, concise points
                       Do not use **(Continous Double Asterisks)
                       Do not only single * anywhere
                       • Include relevant details from the context
                       • Maintain a professional tone without emojis
                       • Keep each point focused and specific

                    3. End with:
                       "Would you like to ask any further questions?"
                       1. [First follow-up question (4-5 words)]
                       2. [Second follow-up question (4-5 words)]

                    Context Documents (choose the most relevant one(s) to answer the query):
                    {full_context}

                    Please answer this query: {content}

                    Remember to:
    - Keep the tone professional and engaging without using any emojis
    - {"Focus on introducing the organization and suggesting relevant questions" if is_greeting else "Provide specific, relevant information"}
    -Strictly follow this for ending the response :
    End with:
                       "Would you like to ask any further questions?"
                       1. [First follow-up question (4-5 words)]
                       2. [Second follow-up question (4-5 words)]"""

                    # Print the final prompt with token count
                    print("\nFinal Prompt Sent to OpenAI:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)

                    # Count tokens (words) in prompt
                    prompt_tokens = len(prompt.split())
                    print(
                        f"\nPrompt Token Count: {prompt_tokens} tokens (words)"
                    )

                    now = datetime.utcnow()

                    messages = []
                    current_input_tokens = prompt_tokens

                    # Check if conversation exists first and handle 5-minute timeout
                    existing_convo = supabase.table('testing_zaps2')\
                        .select('*')\
                        .eq('chatbot_id', user_id)\
                        .eq('session_id', session_id)\
                        .execute()

                    should_create_new = False
                    if existing_convo.data:
                        # Check if conversation is older than 5 minutes
                        try:
                            from datetime import datetime, timedelta
                            created_at_str = existing_convo.data[0].get(
                                'created_at')
                            if created_at_str:
                                created_at = datetime.fromisoformat(
                                    created_at_str.replace('Z', '+00:00'))
                                time_diff = datetime.now(
                                ) - created_at.replace(tzinfo=None)
                                if time_diff > timedelta(minutes=5):
                                    # Modify existing session_id by appending timestamp
                                    current_timestamp = int(time.time())
                                    new_session_id = f"{session_id}_{current_timestamp}"

                                    # Update existing conversation's session_id
                                    supabase.table('testing_zaps2').update({
                                        'session_id':
                                        new_session_id
                                    }).eq('chatbot_id',
                                          user_id).eq('session_id',
                                                      session_id).execute()

                                    should_create_new = True
                        except Exception as e:
                            # If there's any error parsing time, create new conversation
                            should_create_new = True

                    # Only add greeting if creating new conversation and greeting exists
                    if (not existing_convo.data
                            or should_create_new) and greeting_message:
                        messages.append({
                            "role": "assistant",
                            "content": greeting_message
                        })

                    messages.append({"role": "user", "content": content})

                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }])
                    chat_response = completion.choices[0].message.content

                    # Count tokens in response
                    response_tokens = len(chat_response.split())
                    print(
                        f"\nResponse Token Count: {response_tokens} tokens (words)"
                    )

                    messages.append({
                        "role": "assistant",
                        "content": chat_response
                    })

                    if existing_convo.data and not should_create_new:
                        # Update existing conversation
                        existing_row = existing_convo.data[0]
                        existing_messages = existing_row.get('messages', [])
                        updated_messages = existing_messages + messages

                        # Update token counts
                        input_tokens = (existing_row.get('input_tokens')
                                        or 0) + current_input_tokens
                        output_tokens = (existing_row.get('output_tokens')
                                         or 0) + response_tokens

                        supabase.table('testing_zaps2').update({
                            'messages':
                            updated_messages,
                            'input_tokens':
                            input_tokens,
                            'output_tokens':
                            output_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat()
                        }).eq('chatbot_id',
                              user_id).eq('session_id', session_id).execute()
                    else:
                        # Create new conversation
                        supabase.table('testing_zaps2').insert({
                            'chatbot_id':
                            user_id,
                            'session_id':
                            session_id,
                            'messages':
                            messages,
                            'input_tokens':
                            current_input_tokens,
                            'output_tokens':
                            response_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat(),
                            'created_at':
                            now.isoformat()
                        }).execute()

                    # Format response for WhatsApp - keep markdown tables and use proper line breaks
                    if '|' in chat_response and '-' in chat_response:
                        # Keep markdown table format for WhatsApp
                        # WhatsApp supports basic markdown table formatting
                        chat_response = chat_response  # Keep original markdown table format

                    # Ensure proper WhatsApp formatting:
                    # 1. Use \n for line breaks (WhatsApp native)
                    # 2. Bold text with *text*
                    # 3. Italic text with _text_
                    # 4. Keep bullet points as • or -

                    # Replace HTML breaks with actual line breaks
                    chat_response = chat_response.replace('<br>', '\n')

                    # Format bullet points for better WhatsApp display
                    chat_response = chat_response.replace('•', '• ')

                    # Ensure double line breaks for paragraph separation
                    chat_response = chat_response.replace('\n\n', '\n\n')

                    # Clean up any remaining HTML tags that might interfere
                    import re
                    chat_response = re.sub(r'<[^>]+>', '', chat_response)
                    timing_info['chat_response_generation'] = time.time(
                    ) - chat_response_start

                except Exception as e:
                    print(f"Chat response generation failed: {str(e)}")
                    timing_info['chat_response_generation'] = 0.0

        response_start = time.time()
        response_data = {
            "status":
            "success",
            "most_similar_content": {
                "content":
                most_similar_doc["content"] if most_similar_doc else None,
                "similarity":
                similarities[1]["similarity"]
                if len(similarities) > 1 else None,
                "metadata":
                most_similar_doc["metadata"] if most_similar_doc else None
            },
            "chat_response":
            chat_response,
            "query_embedding_length":
            len(embedding),
            "all_documents": [{
                "id":
                doc["id"],
                "first_5_embedding":
                doc["embedding"][:5] if isinstance(doc["embedding"], list) else
                json.loads(doc["embedding"])[:5],
                "embedding_length":
                len(doc["embedding"]) if isinstance(doc["embedding"], list)
                else len(json.loads(doc["embedding"])),
                "metadata":
                doc["metadata"],
                "similarity":
                next((s["similarity"]
                      for s in similarities if s["id"] == doc["id"]), 0)
            } for doc in all_docs.data],
            "similarities":
            similarities,
            "timing":
            timing_info  # Added timing information to response
        }
        timing_info['response_preparation'] = time.time() - response_start

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Processing failed: {str(e)}")


###################################
@app.post("/process-data-old", response_model=ProcessDataResponse)
async def process_data_old(request: ProcessDataRequest) -> ProcessDataResponse:
    timing_info = {}  # Added to store timing information

    """Process text and store in Supabase with content and embedding using native vector search"""
    try:
        # Generate embedding using OpenAI
        embedding_start = time.time()
        embedding = await generate_embedding(request.content)
        timing_info['embedding_generation'] = time.time() - embedding_start

        # Use Supabase native vector search for better performance
        vector_search_start = time.time()
        
        # Build the RPC call for vector similarity search
        rpc_params = {
            'query_embedding': embedding,
            'match_threshold': 0.1,  # Minimum similarity threshold
            'match_count': 50  # Maximum number of results
        }
        
        # Add chatbot_id filter if provided
        if request.chatbot_id:
            rpc_params['filter_chatbot_id'] = request.chatbot_id
            
        # Call Supabase RPC function for vector similarity search
        # Note: This requires a custom RPC function in Supabase
        try:
            # For now, we'll use a hybrid approach with native similarity search
            query_builder = supabase.table('documents').select(
                'id, content, metadata, embedding, source_embedding, keyword_embedding, chatbot_id'
            )
            
            # Filter by chatbot_id if provided
            if request.chatbot_id:
                query_builder = query_builder.eq('chatbot_id', request.chatbot_id)
                
            # Filter out query-type documents
            query_builder = query_builder.neq('metadata->>type', 'query')
            
            # Execute the query
            all_docs = query_builder.execute()
            
            # Calculate similarities using vectorized operations for better performance
            import numpy as np
            
            similarities = []
            
            if all_docs.data:
                # Convert query embedding to numpy array
                query_vec = np.array(embedding)
                
                for doc in all_docs.data:
                    try:
                        # Get embeddings and convert to numpy arrays
                        content_embedding = np.array(doc["embedding"] if isinstance(
                            doc["embedding"], list) else json.loads(doc["embedding"]))
                        source_embedding = np.array(doc["source_embedding"] if isinstance(
                            doc["source_embedding"], list) else json.loads(doc["source_embedding"]))
                        keyword_embedding = np.array(doc["keyword_embedding"] if isinstance(
                            doc["keyword_embedding"], list) else json.loads(doc["keyword_embedding"]))
                        
                        # Vectorized cosine similarity calculation
                        content_score = np.dot(query_vec, content_embedding) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(content_embedding))
                        source_score = np.dot(query_vec, source_embedding) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(source_embedding))
                        keyword_score = np.dot(query_vec, keyword_embedding) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(keyword_embedding))
                        
                        # Calculate weighted final score
                        final_score = (0.6 * content_score) + (0.2 * source_score) + (0.2 * keyword_score)
                        
                        similarities.append({
                            "id": doc["id"],
                            "similarity": float(final_score),
                            "metadata": doc["metadata"],
                            "scores": {
                                "content": float(content_score),
                                "source": float(source_score),
                                "keyword": float(keyword_score),
                                "final": float(final_score)
                            }
                        })
                    except Exception as e:
                        print(f"Error processing document {doc['id']}: {str(e)}")
                        continue
                
                # Sort by similarity score (numpy is faster for sorting)
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            # Fallback to previous method if native search fails
            raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")
            
        timing_info['vector_search'] = time.time() - vector_search_start

        # Get content of top 3 most similar documents
        top_similar_docs = []
        chat_response = None
        most_similar_doc = None

        if similarities and similarities[0]["similarity"] < 0.35:
            # Get fallback message from chatbot_themes
            if request.chatbot_id:
                fallback_result = supabase.table('chatbot_themes').select(
                    'fallback_message').eq('chatbot_id', request.chatbot_id).execute()
                if fallback_result.data and fallback_result.data[0] and fallback_result.data[0].get('fallback_message'):
                    chat_response = fallback_result.data[0]['fallback_message']
                else:
                    chat_response = "Oops! I am not able to get context to answer your query!"
        else:
            if similarities:  # Use the top 3 most similar documents
                # Get top 3 documents efficiently
                for i in range(min(3, len(similarities))):
                    doc_id = similarities[i]["id"]
                    doc = next((doc for doc in all_docs.data if doc["id"] == doc_id), None)
                    if doc:
                        top_similar_docs.append(doc)

                # Keep the first document for backward compatibility
                most_similar_doc = top_similar_docs[0] if top_similar_docs else None

        chat_response_start = time.time()

        # Only proceed with Gemini response if similarity >= 0.35 and no fallback message set
        if not chat_response and most_similar_doc and request.content:
            # Generate chat response if similar document is found using decomposed helper functions
            
            # Step 1: Manage conversation history
            conversation_context = await _manage_conversation_history(request.user_id, request.session_id)
            
            # Step 2: Generate LLM prompt
            prompt = await _generate_llm_prompt(
                request.content, 
                conversation_context, 
                top_similar_docs, 
                request.chatbot_id
            )
            
            # Step 3: Generate chatbot response
            chat_response, input_tokens, output_tokens = await _generate_chatbot_response(
                prompt, 
                request.user_id, 
                request.session_id
            )
            try:
                    # Fetch theme settings from chatbot_themes
                    additional_instruction = ""
                    response_style = ""
                    greeting_message = ""
                    if chatbot_id:
                        theme_result = supabase.table('chatbot_themes').select(
                            'instruction,response_tone,response_length,greeting'
                        ).eq('chatbot_id', chatbot_id).execute()

                        if theme_result.data and theme_result.data[0]:
                            # Get instruction if present
                            if theme_result.data[0].get('instruction'):
                                additional_instruction = f"\nAdditional Instructions: {theme_result.data[0]['instruction']}\n"

                            if theme_result.data[0].get('greeting'):
                                greeting_message = f"{theme_result.data[0]['greeting']}"

                            # Add tone and length settings if present
                            tone = theme_result.data[0].get('response_tone')
                            length = theme_result.data[0].get(
                                'response_length')

                            if tone or length:
                                response_style = "\nResponse Style Requirements:"
                                if tone:
                                    response_style += f"\n- Maintain a {tone} tone throughout the response"
                                if length:
                                    word_limit = "50-60" if length == "concise" else "100-110"
                                    response_style += f"\n- Keep response within {word_limit} words"

                    # Check if the query is a greeting
                    greetings = [
                        'hi', 'hello', 'hey', 'good morning', 'good afternoon',
                        'good evening', 'greetings'
                    ]
                    is_greeting = content.lower().strip() in greetings

                    # Build response style requirements
                    style_requirements = []
                    if tone := theme_result.data[0].get('response_tone'):
                        style_requirements.append(
                            f"maintain a {tone} tone throughout")
                    if length := theme_result.data[0].get('response_length'):
                        word_limit = "50-60" if length == "concise" else "100-110"
                        style_requirements.append(
                            f"keep response within {word_limit} words")

                    style_instructions = "\n- ".join(
                        style_requirements) if style_requirements else ""

                    # Build context from top 3 documents
                    context_sections = []
                    for i, doc in enumerate(top_similar_docs[:3], 1):
                        # Get document source name from metadata
                        metadata = doc.get('metadata', {})
                        if metadata.get('filename'):
                            source_name = metadata['filename']
                        elif metadata.get('url'):
                            source_name = metadata['url']
                        elif metadata.get('title'):
                            source_name = metadata['title']
                        else:
                            source_name = f"Document {i}"

                        context_sections.append(
                            f"Source: {source_name}\n{doc['content']}")

                    full_context = "\n\n".join(context_sections)

                    prompt = f"""You are an AI assistant representing an organization.{additional_instruction}

                    Previous Conversation:
                    {conversation_context}

                    Response Requirements:
                    - {style_instructions}
                    - Response Requirements:
                    Avoid using emojis, emoticons, or informal shorthand (e.g., "LOL", "btw").

                    Respond in the same language the user used to begin the conversation or the language used during the conversation:

                    If the user uses "Hinglish," respond in "Hindi."
                        Keep responses relevant to the conversation context, connecting with prior interactions to maintain a natural flow.

                        Ensure consistency across responses—avoid contradictions or conflicting information.

                        Structure your responses clearly and logically:

                        Use markdown formatting for clarity:

                        Bold headings using asterisks (e.g., Heading).

                        Bullet points for clear, focused information.

                        Avoid AI-like introductory phrases (e.g., “Certainly,” “Absolutely,” “I’d be happy to”)—keep the tone conversational and natural.

                        Review the document source names below and select the most relevant ones to answer the user’s query, applying the correct context and information from the sources.

                        Crucially:Do not use * anywhere no need to put any text in bold


                    - Remember replying to user in same language as the user asked or conversated in the conversation.
                    - Connect responses to previous conversation context when relevant
                    - Maintain information consistency across queries
                    - Structure response in clear, focused points
                    -Make sure to keep the responses structured and make the headings bold and the points in bullet points in formatting which can be used in markdown format,do not use * anywhere no need to put any text in bold.
                    - Do NOT start responses with AI-like phrases such as "Certainly", "Absolutely", "I'd be happy to", etc. Keep responses natural and conversational
                    - First look at the document source names below and identify which document(s) are most relevant to answer the user query. 
                    -MOST IMPORTANT:Do not use * anywhere no need to put any text in bold.

                                        {"IF THIS IS A GREETING AND NO PREVIOUS CONVERSATION:" if is_greeting and not conversation_messages else ""}
                                        1. {"Start with a warm welcome using first-person plural pronouns (we, our, us). Don't give much detail of organization." if is_greeting and not conversation_messages else "Maintain a natural conversation flow, as this is a continuing discussion."}
                                           {"Focus on giving possible questions users might ask, keeping them relevant to educational institute enquiries only." if is_greeting and not conversation_messages else ""}

                                        2. {"If this is not a greeting, provide information as bullet points:" if not is_greeting else "Suggest possible areas of interest:"}
                                           • Use clear, concise points
                                           • Include relevant details from the context
                                           • Maintain a professional tone without emojis
                                           • Keep each point focused and specific

                                       
                                        IMPORTANT: 
                                        - DO NOT include "Would you like to ask any further questions?" in the response output
                                        - Always generate exactly 2 numbered follow-up questions that are:
                                        - Directly related to the current topic
                                        - Different from each other 
                                        - Actionable and specific
                                        - Between 4-5 words each
                                        -End explicitly with :
                       Would you like to ask any further questions?
                       1. [First follow-up question (4-5 words)]
                       2. [Second follow-up question (4-5 words)]
                       
                                        

                                        Context Documents (choose the most relevant one(s) to answer the query):
                                        {full_context}

                                        Please answer this query: {content}

                                        Remember to:
                        - Keep the tone professional and engaging without using any emojis
                        - {"Focus on introducing the organization and suggesting relevant questions" if is_greeting else "Provide specific, relevant information"}
                        - ALWAYS end with exactly 2 numbered follow-up questions
                        - Make questions 8-12 words each and topic-specific
                        - Ensure questions are different from each other and actionable
                        - Focus on educational institute context and current topic
                        - Do not use * anywhere no need to put any text in bold
                        -Do not say Hello and all everytime if earlier in the previous messages you have sent hello or any greeting so no need to say again ok, take reference from the conversational history provided.
                        -Strictly follow this for ending the response :
    End with:
                       "Would you like to ask any further questions?"
                       1. [First follow-up question (4-5 words)]
                       2. [Second follow-up question (4-5 words)]"""

                    # Print the final prompt with token count
                    print("\nFinal Prompt Sent to Gemini:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)

                    # Count tokens (words) in prompt
                    prompt_tokens = len(prompt.split())
                    print(
                        f"\nPrompt Token Count: {prompt_tokens} tokens (words)"
                    )

                    now = datetime.utcnow()

                    messages = []
                    current_input_tokens = prompt_tokens

                    # Import datetime at the beginning for proper scope
                    from datetime import datetime as dt, timedelta

                    # Check if conversation exists first and handle 5-minute timeout
                    existing_convo = supabase.table('testing_zaps2')\
                        .select('*')\
                        .eq('chatbot_id', user_id)\
                        .eq('session_id', session_id)\
                        .execute()

                    should_create_new = False
                    if existing_convo.data:
                        # Check if conversation is older than 5 minutes
                        try:
                            created_at_str = existing_convo.data[0].get(
                                'created_at')
                            if created_at_str:
                                created_at = dt.fromisoformat(
                                    created_at_str.replace('Z', '+00:00'))
                                time_diff = dt.now() - created_at.replace(
                                    tzinfo=None)
                                if time_diff > timedelta(minutes=30):
                                    # Modify existing session_id by appending timestamp
                                    current_timestamp = int(time.time())
                                    new_session_id = f"{session_id}_{current_timestamp}"

                                    # Update existing conversation's session_id
                                    supabase.table('testing_zaps2').update({
                                        'session_id':
                                        new_session_id
                                    }).eq('chatbot_id',
                                          user_id).eq('session_id',
                                                      session_id).execute()

                                    should_create_new = True
                        except Exception as e:
                            # If there's any error parsing time, create new conversation
                            should_create_new = True

                    # Only add greeting if creating new conversation and greeting exists
                    if (not existing_convo.data
                            or should_create_new) and greeting_message:
                        messages.append({
                            "role": "assistant",
                            "content": greeting_message
                        })

                    messages.append({"role": "user", "content": content})

                    # Use Gemini instead of OpenAI
                    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    response = model.generate_content(prompt)
                    chat_response = response.text

                    # Validate and fix response formatting
                    chat_response, formatting_issues = validate_response_formatting(
                        chat_response)
                    if formatting_issues:
                        print(f"Fixed formatting issues: {formatting_issues}")

                    # Count tokens in response
                    response_tokens = len(chat_response.split())
                    print(
                        f"\nResponse Token Count: {response_tokens} tokens (words)"
                    )

                    messages.append({
                        "role": "assistant",
                        "content": chat_response
                    })

                    if existing_convo.data and not should_create_new:
                        # Update existing conversation
                        existing_row = existing_convo.data[0]
                        existing_messages = existing_row.get('messages', [])
                        updated_messages = existing_messages + messages

                        # Update token counts
                        input_tokens = (existing_row.get('input_tokens')
                                        or 0) + current_input_tokens
                        output_tokens = (existing_row.get('output_tokens')
                                         or 0) + response_tokens

                        supabase.table('testing_zaps2').update({
                            'messages':
                            updated_messages,
                            'input_tokens':
                            input_tokens,
                            'output_tokens':
                            output_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat()
                        }).eq('chatbot_id',
                              user_id).eq('session_id', session_id).execute()
                    else:
                        # Create new conversation
                        supabase.table('testing_zaps2').insert({
                            'chatbot_id':
                            user_id,
                            'session_id':
                            session_id,
                            'messages':
                            messages,
                            'input_tokens':
                            current_input_tokens,
                            'output_tokens':
                            response_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat(),
                            'created_at':
                            now.isoformat()
                        }).execute()

                    # Apply comprehensive text formatting and validation
                    chat_response = format_chatbot_response(chat_response)
            
                    timing_info['chat_response_generation'] = time.time() - chat_response_start


        # Prepare response data with proper typing
        response_start = time.time()
        
        # Format all documents response
        formatted_documents = []
        for doc in all_docs.data:
            embedding_data = doc["embedding"] if isinstance(doc["embedding"], list) else json.loads(doc["embedding"])
            doc_similarity = next((s["similarity"] for s in similarities if s["id"] == doc["id"]), 0.0)
            
            formatted_documents.append(DocumentResponse(
                id=doc["id"],
                first_5_embedding=embedding_data[:5],
                embedding_length=len(embedding_data),
                metadata=doc["metadata"],
                similarity=doc_similarity
            ))
        
        # Format similarities response
        formatted_similarities = [
            SimilarityScore(
                id=sim["id"],
                similarity=sim["similarity"],
                metadata=sim["metadata"],
                scores=sim.get("scores")
            ) for sim in similarities
        ]
        
        timing_info['response_preparation'] = time.time() - response_start

        return ProcessDataResponse(
            status="success",
            most_similar_content={
                "content": most_similar_doc["content"] if most_similar_doc else None,
                "similarity": similarities[0]["similarity"] if similarities else None,
                "metadata": most_similar_doc["metadata"] if most_similar_doc else None
            },
            chat_response=chat_response,
            query_embedding_length=len(embedding),
            all_documents=formatted_documents,
            similarities=formatted_similarities,
            timing=timing_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


##################################
@app.post("/process-data/whatsapp")
async def process_data_whastapp(request: Dict):
    # Validate required parameters
    content = request.get("content")
    user_id = request.get("user_id")
    session_id = request.get("session_id")
    if not all([content, user_id, session_id]):
        raise HTTPException(
            status_code=422,
            detail=
            "Missing required parameters: content, user_id, and session_id are mandatory"
        )

    metadata = request.get("metadata", {})
    chatbot_id = request.get("chatbot_id")
    timing_info = {}  #Added to store timing information

    # Log user message first

    # supabase.table('chat_messages').insert(user_message_data).execute()
    """Process text and store in Supabase with content and embedding"""
    try:
        # Generate embedding using OpenAI
        embedding_start = time.time()
        embedding = await generate_embedding(content)
        timing_info['embedding_generation'] = time.time() - embedding_start

        document_search_start = time.time()

        # Fetch documents filtered by chatbot_id
        query = supabase.table('documents').select('*')
        if chatbot_id:
            query = query.eq('chatbot_id', chatbot_id)
        all_docs = query.execute()
        timing_info['fetch_all_docs'] = time.time() - document_search_start

        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Calculate similarities with all documents using all embedding types
        similarities = []
        cosine_similarity_start = time.time()

        for doc in all_docs.data:
            # Skip documents that are queries
            if doc.get("metadata", {}).get("type") == "query":
                continue

            # Get content embedding
            content_embedding = doc["embedding"] if isinstance(
                doc["embedding"], list) else json.loads(doc["embedding"])
            content_score = cosine_similarity(embedding, content_embedding)

            # Get source embedding
            source_embedding = doc["source_embedding"] if isinstance(
                doc["source_embedding"], list) else json.loads(
                    doc["source_embedding"])
            source_score = cosine_similarity(embedding, source_embedding)

            # Get keyword embedding
            keyword_embedding = doc["keyword_embedding"] if isinstance(
                doc["keyword_embedding"], list) else json.loads(
                    doc["keyword_embedding"])
            keyword_score = cosine_similarity(embedding, keyword_embedding)

            # Calculate final weighted score
            final_score = (0.6 * content_score) + (0.2 * source_score) + (
                0.2 * keyword_score)

            similarities.append({
                "id": doc["id"],
                "similarity": float(final_score),
                "metadata": doc["metadata"],
                "scores": {
                    "content": float(content_score),
                    "source": float(source_score),
                    "keyword": float(keyword_score),
                    "final": float(final_score)
                }
            })

        # Sort by final similarity score
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Get content of top 3 most similar documents
        top_similar_docs = []
        chat_response = None

        if similarities and similarities[0]["similarity"] < 0.35:
            # Get fallback message from chatbot_themes
            if chatbot_id:
                fallback_result = supabase.table('chatbot_themes').select(
                    'fallback_message').eq('chatbot_id', chatbot_id).execute()
                if fallback_result.data and fallback_result.data[
                        0] and fallback_result.data[0].get('fallback_message'):
                    chat_response = fallback_result.data[0]['fallback_message']
                else:
                    chat_response = "Oops! I am not able to get context to answer your query!"
        else:
            if similarities:  # Use the top 3 most similar documents
                # Get top 3 documents
                for i in range(min(3, len(similarities))):
                    doc_id = similarities[i]["id"]
                    doc = next(
                        (doc for doc in all_docs.data if doc["id"] == doc_id),
                        None)
                    if doc:
                        top_similar_docs.append(doc)

                # Keep the first document for backward compatibility
                most_similar_doc = top_similar_docs[
                    0] if top_similar_docs else None

        timing_info['cosine_similarity_process'] = time.time(
        ) - cosine_similarity_start
        chat_response_start = time.time()

        # Only proceed with OpenAI response if similarity >= 0.65 and no fallback message set
        if not chat_response:
            # Generate chat response if similar document is found

            # Get previous conversation history
            conversation_history = supabase.table('chat_messages')\
                .select('*')\
                .eq('user_id', user_id)\
                .eq('session_id', session_id)\
                .order('timestamp', desc=True)\
                .limit(6)\
                .execute()  # Get last 6 messages (3 pairs of user-assistant interactions)

            # Format conversation history in chronological order
            conversation_messages = []
            if conversation_history.data:
                # Reverse to get chronological order and format messages
                for msg in reversed(conversation_history.data):
                    role = "User" if msg[
                        'message_type'] == 'user' else "Assistant"
                    conversation_messages.append(f"{role}: {msg['message']}")

            # Convert to string with newlines between messages
            conversation_context = "\n".join(
                conversation_messages) if conversation_messages else ""

            if most_similar_doc and content:  # content here is the query
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    # Fetch theme settings from chatbot_themes
                    additional_instruction = ""
                    response_style = ""
                    greeting_message = ""
                    if chatbot_id:
                        theme_result = supabase.table('chatbot_themes').select(
                            'instruction,response_tone,response_length,greeting'
                        ).eq('chatbot_id', chatbot_id).execute()

                        if theme_result.data and theme_result.data[0]:
                            # Get instruction if present
                            if theme_result.data[0].get('instruction'):
                                additional_instruction = f"\nAdditional Instructions: {theme_result.data[0]['instruction']}\n"

                            if theme_result.data[0].get('greeting'):
                                greeting_message = f"{theme_result.data[0]['greeting']}"

                            # Add tone and length settings if present
                            tone = theme_result.data[0].get('response_tone')
                            length = theme_result.data[0].get(
                                'response_length')

                            if tone or length:
                                response_style = "\nResponse Style Requirements:"
                                if tone:
                                    response_style += f"\n- Maintain a {tone} tone throughout the response"
                                if length:
                                    word_limit = "50-60" if length == "concise" else "100-110"
                                    response_style += f"\n- Keep response within {word_limit} words"

                    # Check if the query is a greeting
                    greetings = [
                        'hi', 'hello', 'hey', 'good morning', 'good afternoon',
                        'good evening', 'greetings'
                    ]
                    is_greeting = content.lower().strip() in greetings

                    # Build response style requirements
                    style_requirements = []
                    if tone := theme_result.data[0].get('response_tone'):
                        style_requirements.append(
                            f"maintain a {tone} tone throughout")
                    if length := theme_result.data[0].get('response_length'):
                        word_limit = "50-60" if length == "concise" else "100-110"
                        style_requirements.append(
                            f"keep response within {word_limit} words")

                    style_instructions = "\n- ".join(
                        style_requirements) if style_requirements else ""

                    # Build context from top 3 documents
                    context_sections = []
                    for i, doc in enumerate(top_similar_docs[:3], 1):
                        # Get document source name from metadata
                        metadata = doc.get('metadata', {})
                        if metadata.get('filename'):
                            source_name = metadata['filename']
                        elif metadata.get('url'):
                            source_name = metadata['url']
                        elif metadata.get('title'):
                            source_name = metadata['title']
                        else:
                            source_name = f"Document {i}"

                        context_sections.append(
                            f"Source: {source_name}\n{doc['content']}")

                    full_context = "\n\n".join(context_sections)

                    prompt = f"""You are an AI assistant representing an organization.{additional_instruction}

Previous Conversation:
{conversation_context}

Response Requirements:
- {style_instructions}
- Do not use emojis or emoticons
- Connect responses to previous conversation context when relevant
- Maintain information consistency across queries
- Structure response in clear, focused points
- IMPORTANT: Use ONLY "\n" for line breaks and "*" for bullet points. Do NOT use any HTML tags, special characters, or other markdown symbols as this is for WhatsApp formatting
- Also, do not use "/" this slash symbol also in markdown formatting purely use only the \n and * nothing else
- Do NOT start responses with AI-like phrases such as "Certainly", "Absolutely", "I'd be happy to", etc. Keep responses natural and conversational
- First look at the document source names below and identify which document(s) are most relevant to answer the user query

                    {"IF THIS IS A GREETING AND NO PREVIOUS CONVERSATION:" if is_greeting and not conversation_messages else ""}
                    1. {"Start with a warm welcome using first-person plural pronouns (we, our, us). Don't give much detail of organization." if is_greeting and not conversation_messages else "Maintain a natural conversation flow, as this is a continuing discussion."}
                       {"Focus on giving possible questions users might ask, keeping them relevant to educational institute enquiries only." if is_greeting and not conversation_messages else ""}

                    2. {"If this is not a greeting, provide information as bullet points using only asterisk (*) followed by space:" if not is_greeting else "Suggest possible areas of interest using only asterisk (*) followed by space:"}
                       * Use clear, concise points
                       * Include relevant details from the context
                       * Maintain a professional tone without emojis
                       * Keep each point focused and specific

                    3. End with follow-up suggestions in this EXACT format:
                       
                       1. [Write a specific question related to the topic - 4-5 words]
                       2. [Write another specific question about related aspects - 4-5 words]
                       
                       IMPORTANT: 
                       - DO NOT include "Would you like to ask any further questions?" in the response output
                       - Always generate exactly 2 numbered follow-up questions that are:
                       - Directly related to the current topic
                       - Different from each other
                       - Actionable and specific  
                       - Between 4-5 words each

                    FORMATTING RULES FOR WHATSAPP:
                    - Use "\n" for single line breaks
                    - Use "\n\n" for paragraph breaks
                    - Use "*" followed by space for bullet points (e.g., "* Point here")
                    - Use numbered lists with "1. ", "2. " etc.
                    - NO HTML tags, NO special symbols, NO other markdown formatting

                    Example format:
                    "Response text here.\n\n* First bullet point\n* Second bullet point\n\nWould you like to ask any further questions?\n\n1. Question one\n2. Question two"

                    Context Documents (choose the most relevant one(s) to answer the query):
                    {full_context}

                    Please answer this query: {content}

                    Remember to:
    - Keep the tone professional and engaging without using any emojis
    - {"Focus on introducing the organization and suggesting relevant questions" if is_greeting else "Provide specific, relevant information"}
    - ALWAYS end with exactly 2 numbered follow-up questions
    - Make questions 8-12 words each and topic-specific
    - Ensure questions are different from each other and actionable
    - Focus on educational institute context and current topic
    - Use ONLY "\\n" and "*" for formatting - no other symbols or tags
    - Start responses naturally without AI-like phrases
"""

                    # Print the final prompt with token count
                    print("\nFinal Prompt Sent to OpenAI:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)

                    # Count tokens (words) in prompt
                    prompt_tokens = len(prompt.split())
                    print(
                        f"\nPrompt Token Count: {prompt_tokens} tokens (words)"
                    )

                    now = datetime.utcnow()

                    messages = []
                    current_input_tokens = prompt_tokens

                    # Import datetime at the beginning for proper scope
                    from datetime import datetime as dt, timedelta

                    # Check if conversation exists first and handle 5-minute timeout
                    existing_convo = supabase.table('testing_zaps2')\
                        .select('*')\
                        .eq('chatbot_id', user_id)\
                        .eq('session_id', session_id)\
                        .execute()

                    should_create_new = False
                    if existing_convo.data:
                        # Check if conversation is older than 5 minutes
                        try:
                            created_at_str = existing_convo.data[0].get(
                                'created_at')
                            if created_at_str:
                                created_at = dt.fromisoformat(
                                    created_at_str.replace('Z', '+00:00'))
                                time_diff = dt.now() - created_at.replace(
                                    tzinfo=None)
                                if time_diff > timedelta(minutes=30):
                                    # Modify existing session_id by appending timestamp
                                    current_timestamp = int(time.time())
                                    new_session_id = f"{session_id}_{current_timestamp}"

                                    # Update existing conversation's session_id
                                    supabase.table('testing_zaps2').update({
                                        'session_id':
                                        new_session_id
                                    }).eq('chatbot_id',
                                          user_id).eq('session_id',
                                                      session_id).execute()

                                    should_create_new = True
                        except Exception as e:
                            # If there's any error parsing time, create new conversation
                            should_create_new = True

                    # Only add greeting if creating new conversation and greeting exists
                    if (not existing_convo.data
                            or should_create_new) and greeting_message:
                        messages.append({
                            "role": "assistant",
                            "content": greeting_message
                        })

                    messages.append({"role": "user", "content": content})

                    # Use Gemini instead of OpenAI
                    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    response = model.generate_content(prompt)
                    chat_response = response.text

                    # Validate and fix response formatting
                    chat_response, formatting_issues = validate_response_formatting(
                        chat_response)
                    if formatting_issues:
                        print(f"Fixed formatting issues: {formatting_issues}")

                    # Count tokens in response
                    response_tokens = len(chat_response.split())
                    print(
                        f"\nResponse Token Count: {response_tokens} tokens (words)"
                    )

                    messages.append({
                        "role": "assistant",
                        "content": chat_response
                    })

                    if existing_convo.data and not should_create_new:
                        # Update existing conversation
                        existing_row = existing_convo.data[0]
                        existing_messages = existing_row.get('messages', [])
                        updated_messages = existing_messages + messages

                        # Update token counts
                        input_tokens = (existing_row.get('input_tokens')
                                        or 0) + current_input_tokens
                        output_tokens = (existing_row.get('output_tokens')
                                         or 0) + response_tokens

                        supabase.table('testing_zaps2').update({
                            'messages':
                            updated_messages,
                            'input_tokens':
                            input_tokens,
                            'output_tokens':
                            output_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat()
                        }).eq('chatbot_id',
                              user_id).eq('session_id', session_id).execute()
                    else:
                        # Create new conversation
                        supabase.table('testing_zaps2').insert({
                            'chatbot_id':
                            user_id,
                            'session_id':
                            session_id,
                            'messages':
                            messages,
                            'input_tokens':
                            current_input_tokens,
                            'output_tokens':
                            response_tokens,
                            'date_of_convo':
                            now.date().isoformat(),
                            'time_of_convo':
                            now.time().isoformat(),
                            'created_at':
                            now.isoformat()
                        }).execute()

                    # For WhatsApp, keep the response as plain text with only \n and * formatting
                    # Remove any HTML tags or special formatting that might interfere
                    import re
                    # Clean up any HTML tags
                    chat_response = re.sub(r'<[^>]+>', '', chat_response)

                    # Ensure proper WhatsApp formatting by replacing any incorrect formatting
                    # Convert bullet points to asterisk format if needed
                    chat_response = re.sub(r'•\s*', '* ', chat_response)
                    chat_response = re.sub(r'-\s*', '* ', chat_response)

                    # Ensure line breaks are clean
                    chat_response = chat_response.replace('<br>', '\n')
                    chat_response = chat_response.replace('<br/>', '\n')
                    chat_response = chat_response.replace('<br />', '\n')

                    timing_info['chat_response_generation'] = time.time(
                    ) - chat_response_start

                except Exception as e:
                    print(f"Chat response generation failed: {str(e)}")
                    timing_info['chat_response_generation'] = 0.0

        response_start = time.time()
        response_data = {
            "status":
            "success",
            "most_similar_content": {
                "content":
                most_similar_doc["content"] if most_similar_doc else None,
                "similarity":
                similarities[1]["similarity"]
                if len(similarities) > 1 else None,
                "metadata":
                most_similar_doc["metadata"] if most_similar_doc else None
            },
            "chat_response":
            chat_response,
            "query_embedding_length":
            len(embedding),
            "all_documents": [{
                "id":
                doc["id"],
                "first_5_embedding":
                doc["embedding"][:5] if isinstance(doc["embedding"], list) else
                json.loads(doc["embedding"])[:5],
                "embedding_length":
                len(doc["embedding"]) if isinstance(doc["embedding"], list)
                else len(json.loads(doc["embedding"])),
                "metadata":
                doc["metadata"],
                "similarity":
                next((s["similarity"]
                      for s in similarities if s["id"] == doc["id"]), 0)
            } for doc in all_docs.data],
            "similarities":
            similarities,
            "timing":
            timing_info  # Added timing information to response
        }
        timing_info['response_preparation'] = time.time() - response_start

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Processing failed: {str(e)}")


@app.post("/knowledge_url_upload/")
async def knowledge_url_upload(urls: List[str],
                               chatbot_id: str,
                               metadata: Dict = {}):
    """Handle URL crawling with automatic embedding generation for each page"""
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    if not chatbot_id:
        raise HTTPException(status_code=400, detail="chatbot_id is required")
    if not isinstance(urls, list):
        raise HTTPException(status_code=400,
                            detail="URLs must be provided as a list")

    async def event_stream():
        try:
            print(f"Processing URLs: {urls}")
            print(f"Chatbot ID: {chatbot_id}")
            print(f"Metadata: {metadata}")

            crawler = WebCrawler()
            results = []

            async for data in crawler.crawl_pages(urls):
                yield f"data: {json.dumps(data)}\n\n"

                if data["status"] == "complete":
                    # Process crawled pages
                    filename = data["download_url"].split("/")[-1]
                    with open(filename, 'r', encoding='utf-8') as f:
                        crawl_data = json.load(f)

                        for url, page_data in crawl_data["pages"].items():
                            if page_data["status"] == "success":
                                try:
                                    # Prepare metadata
                                    page_metadata = {
                                        "url": url,
                                        "title": page_data["title"],
                                        "type": "webpage",
                                        "source": "web_crawl",
                                        "crawl_timestamp":
                                        page_data["timestamp"]
                                    }
                                    # Merge with user provided metadata
                                    page_metadata.update(metadata)

                                    # Generate embedding
                                    embedding = await generate_embedding(
                                        page_data["text_content"])

                                    # Store content and embedding as separate entry
                                    data = {
                                        "content":
                                        page_data["text_content"][:80000],
                                        "embedding":
                                        embedding,
                                        "metadata":
                                        page_metadata,
                                        "created_at":
                                        datetime.utcnow().isoisoformat(),
                                        "chatbot_id":
                                        chatbot_id
                                    }

                                    # Insert document as individual row
                                    result = supabase.table(
                                        'documents').insert(data).execute()
                                    if result.data:
                                        results.extend(result.data)
                                        yield f"data: {json.dumps({'status': 'success', 'url': url, 'message': 'Processed and stored successfully'})}\n\n"
                                except Exception as e:
                                    yield f"data: {json.dumps({'status': 'error', 'url': url, 'detail': str(e)})}\n\n"

                # Clean up crawl file
                    os.remove(filename)

                    # Final response
                    yield f"data: {json.dumps({'status': 'embedding_complete', 'processed_pages': len(results)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


#####################################################


@app.post("/knowledge_doc_upload/")
async def knowledge_doc_upload(file: UploadFile,
                               metadata: str = Form("{}"),
                               chatbot_id: str = Form(None)):
    """Handle document upload withautomatic embedding generation"""
    # Parse metadata string to dict
    try:
        metadata = json.loads(metadata)
    except json.JSONDecodeError:
        metadata = {}
    try:
        # Add file metadata automatically
        file_metadata = {
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].lower()
        }
        # Merge with user provided metadata
        metadata.update(file_metadata)

        # Read and extract text from file
        content = await file.read()

        if file.filename.lower().endswith('.pdf'):
            text = await extract_text_from_pdf(content)
        elif file.filename.lower().endswith('.docx'):
            text = await extract_text_from_docx(content)
        elif file.filename.lower().endswith('.json'):
            try:
                json_content = json.loads(content.decode('utf-8'))
                text = json.dumps(json_content, indent=2)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400,
                                    detail="Invalid JSON file")
        else:
            raise HTTPException(status_code=400,
                                detail="Unsupported file format")

        # Generate embedding
        embedding = await generate_embedding(text)

        # Store content and embedding
        data = {
            "content": text[:80000],  # Limiting content length for storage
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
            "chatbot_id": chatbot_id
        }

        # Insert document
        result = supabase.table('documents').insert(data).execute()

        return {
            "status": "success",
            "data": result.data,
            "query_embedding_length": len(embedding)
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Processing failed: {str(e)}")


async def generate_keywords(text: str, metadata: dict = None) -> list:
    """Generate comprehensive list of potential user questions"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Prepare metadata context and determine main contextual word
        main_context = ""
        metadata_context = ""
        if metadata:
            if url := metadata.get('url', ''):
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                path_parts = [p for p in parsed_url.split('/') if p]
                main_context = path_parts[-1] if path_parts else "evonix"
                metadata_context = f"URL Type: {main_context}\nDomain: {parsed_url.netloc}"
            elif filename := metadata.get('filename', ''):
                main_context = filename.split('.')[0].replace('_', ' ').lower()
                metadata_context = f"Document Type: {filename.split('.')[-1]}"

        # Generate unified question prompt
        unified_prompt = f"""Generate 20 diverse questions that users might ask about this content. Include different question types and variations.

SOURCE INFO:
{metadata_context}
Main Context: {main_context}

CONTENT PREVIEW:
{text[:1000]}

REQUIREMENTS:
1. Include various question types:
   - What/How/Why/When/Where/Who questions
   - Yes/No questions
   - Comparative questions
   - Open-ended questions
   - Specific detail questions
2. Focus on the main topic ({main_context})
3. Include questions about:
   - Basic information
   - Detailed specifics
   - Process/methods
   - Benefits/advantages
   - Requirements/prerequisites
   - Comparisons
4. Format as numbered list (1. Question\\n2. Question\\n...)

Example if content is about services:
1. What services does [company] offer?
2. How can I get started with [service]?
3. What are the benefits of [service]?
4. Do you provide [specific feature]?
5. Which service is best for [specific need]?
etc."""

        response = client.chat.completions.create(model="gpt-4",
                                                  messages=[{
                                                      "role":
                                                      "user",
                                                      "content":
                                                      unified_prompt
                                                  }])

        # Split response into individual questions and ensure proper formatting
        questions = response.choices[0].message.content.split('\n')
        questions = [
            q.strip().split('. ', 1)[1] if '. ' in q else q.strip()
            for q in questions if q.strip()
        ]

        return questions[:20]  # Ensure exactly 20 questions are returned
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI keyword generation failed: {str(e)}")


@app.get("/user_doc/")
async def user_doc():
    """Fetch and read documents from Supabase Object Storage, generate embeddings and store in documents table"""
    try:
        # Get list of all folders (chatbot IDs) in documentuploaded bucket
        folders = supabase.storage.from_('documentuploaded').list()

        processed_files = []

        # Iterate through each folder (chatbot ID)
        for folder in folders:
            folder_name = folder['name']  # This is the chatbot_id

            # Get list of files in current folder
            files = supabase.storage.from_('documentuploaded').list(
                folder_name)

            # Process each file in the folder
            for file in files:
                file_name = file['name']
                file_type = file_name.split('.')[-1].lower()

                try:
                    # Check if document already exists in doc_ids table
                    existing_doc = supabase.table('doc_ids').select('*').eq(
                        'document_name', file_name).eq('chatbot_id',
                                                       folder_name).execute()

                    if existing_doc.data:
                        # Document already exists, skip processing
                        processed_files.append({
                            "filename":
                            file_name,
                            "chatbot_id":
                            folder_name,
                            "status":
                            "skipped",
                            "message":
                            "Document already processed",
                            "document_id":
                            existing_doc.data[0]['document_id']
                        })
                        continue

                    # Download file content
                    file_data = supabase.storage.from_(
                        'documentuploaded').download(
                            f"{folder_name}/{file_name}")

                    # Extract text based on file type
                    if file_type == 'pdf':
                        text = await extract_text_from_pdf(file_data)
                    elif file_type == 'docx':
                        text = await extract_text_from_docx(file_data)
                    else:
                        continue  # Skip unsupported file types

                    # Generate keywords and embeddings
                    keywords = await generate_keywords(text,
                                                       {"filename": file_name})
                    keywords_text = ", ".join(keywords)

                    # Generate embeddings for content, keywords and source
                    content_embedding = await generate_embedding(text)
                    keyword_embedding = await generate_embedding(keywords_text)
                    source_embedding = await generate_embedding(
                        file_name)  # Generate embedding for filename

                    # Prepare metadata with file information
                    metadata = {
                        "filename": file_name,
                        "file_type": file_type,
                        "source": "document_upload",
                        "upload_timestamp": datetime.utcnow().isoformat()
                    }

                    # Store in Supabase documents table
                    document_data = {
                        "content": text[:80000],  # Limiting content length
                        "embedding": content_embedding,
                        "metadata": metadata,
                        "created_at": datetime.utcnow().isoformat(),
                        "chatbot_id": folder_name,
                        "keywords": keywords_text,
                        "keyword_embedding": keyword_embedding,
                        "source_embedding": source_embedding
                    }

                    result = supabase.table('documents').insert(
                        document_data).execute()
                    document_id = result.data[0]['id'] if result.data else None

                    if document_id:
                        # Store in doc_ids table
                        doc_id_data = {
                            "document_id": document_id,
                            "document_name": file_name,
                            "chatbot_id": folder_name
                        }
                        supabase.table('doc_ids').insert(doc_id_data).execute()

                        # Check and update chatbot status if needed
                        credentials = supabase.table('credentials').select(
                            'chatbot_status').eq('chatbot_id',
                                                 folder_name).execute()
                        if credentials.data and credentials.data[0][
                                'chatbot_status'] != 'Active':
                            supabase.table('credentials').update({
                                'chatbot_status':
                                'Active'
                            }).eq('chatbot_id', folder_name).execute()

                        processed_files.append({
                            "filename": file_name,
                            "chatbot_id": folder_name,
                            "status": "success",
                            "document_id": document_id
                        })

                except Exception as e:
                    processed_files.append({
                        "filename": file_name,
                        "chatbot_id": folder_name,
                        "status": "error",
                        "error": str(e)
                    })

        return {"status": "success", "processed_files": processed_files}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing documents: {str(e)}")


@app.get("/user_url/")
async def user_url():
    """Fetch and process URLs from urls_uploaded table"""
    try:
        # Fetch all rows from urls_uploaded table
        result = supabase.table('urls_uploaded').select('*').execute()

        if not result.data:
            return {"status": "success", "message": "No URLs found to process"}

        processed_urls = []
        crawler = WebCrawler()

        # Fetch chatbot IDs with refresh = "Yes"
        refresh_result = supabase.table('urls_uploaded').select(
            'chatbot_id, refresh').eq('refresh', 'Yes').execute()
        refresh_chatbots = []

        for row in refresh_result.data:
            chatbot_id = row.get('chatbot_id')
            if chatbot_id:
                refresh_chatbots.append(chatbot_id)
                # Immediately set refresh to "No"
                supabase.table('urls_uploaded').update({
                    'refresh': 'No'
                }).eq('chatbot_id', chatbot_id).execute()

        for row in result.data:
            try:
                url_data = row.get('url_links', {})
                if not url_data or 'links' not in url_data:
                    continue

                chatbot_id = row.get('chatbot_id')
                if not chatbot_id:
                    continue

                urls = url_data['links']

                # Process each URL
                for url in urls:
                    try:
                        document_id = None
                        is_update = False

                        # Only check doc_ids if not in refresh_chatbots
                        if chatbot_id not in refresh_chatbots:
                            existing_doc = supabase.table('doc_ids').select(
                                '*').eq('document_name',
                                        url).eq('chatbot_id',
                                                chatbot_id).execute()

                            if existing_doc.data:
                                # URL already processed, skip it
                                processed_urls.append({
                                    "url":
                                    url,
                                    "chatbot_id":
                                    chatbot_id,
                                    "status":
                                    "skipped",
                                    "message":
                                    "URL already processed",
                                    "document_id":
                                    existing_doc.data[0]['document_id']
                                })
                                continue
                        else:
                            # For refresh cases, get existing document_id
                            existing_doc = supabase.table('doc_ids').select(
                                '*').eq('document_name',
                                        url).eq('chatbot_id',
                                                chatbot_id).execute()
                            if existing_doc.data:
                                document_id = existing_doc.data[0][
                                    'document_id']
                                is_update = True

                        # Use the existing crawl_pages generator
                        async for data in crawler.crawl_pages([url]):
                            if data['status'] == 'complete':
                                # Process the crawled content
                                filename = data['download_url'].split('/')[-1]
                                with open(filename, 'r',
                                          encoding='utf-8') as f:
                                    crawl_data = json.load(f)

                                    for page_url, page_data in crawl_data[
                                            'pages'].items():
                                        if page_data['status'] == 'success':
                                            try:
                                                # Prepare metadata
                                                page_metadata = {
                                                    "url":
                                                    page_url,
                                                    "title":
                                                    page_data["title"],
                                                    "type":
                                                    "webpage",
                                                    "source":
                                                    "url_upload",
                                                    "crawl_timestamp":
                                                    page_data["timestamp"]
                                                }

                                                # Generate keywords
                                                keywords = await generate_keywords(
                                                    page_data["text_content"],
                                                    {"url": page_url})
                                                keywords_text = ", ".join(
                                                    keywords)

                                                # Generate embeddings for content, keywords and source
                                                content_embedding = await generate_embedding(
                                                    page_data["text_content"])
                                                keyword_embedding = await generate_embedding(
                                                    keywords_text)
                                                source_embedding = await generate_embedding(
                                                    page_url
                                                )  # Generate embedding for URL

                                                # Store in Supabase documents table
                                                document_data = {
                                                    "content":
                                                    page_data["text_content"]
                                                    [:80000],
                                                    "embedding":
                                                    content_embedding,
                                                    "metadata":
                                                    page_metadata,
                                                    "created_at":
                                                    datetime.utcnow(
                                                    ).isoformat(),
                                                    "chatbot_id":
                                                    chatbot_id,
                                                    "keywords":
                                                    keywords_text,
                                                    "keyword_embedding":
                                                    keyword_embedding,
                                                    "source_embedding":
                                                    source_embedding
                                                }

                                                if is_update and document_id:
                                                    # Update existing document
                                                    result = supabase.table('documents')\
                                                        .update(document_data)\
                                                        .eq('id', document_id)\
                                                        .execute()
                                                else:
                                                    # Insert new document
                                                    result = supabase.table(
                                                        'documents').insert(
                                                            document_data
                                                        ).execute()
                                                    document_id = result.data[0][
                                                        'id'] if result.data else None

                                                if document_id:
                                                    # Check if document_id already exists in doc_ids
                                                    existing_doc_id = supabase.table('doc_ids').select('*')\
                                                        .eq('document_id', document_id)\
                                                        .execute()

                                                    if existing_doc_id.data:
                                                        # Update created_at for existing document
                                                        supabase.table('doc_ids')\
                                                            .update({'created_at': datetime.utcnow().isoformat()})\
                                                            .eq('document_id', document_id)\
                                                            .execute()
                                                    else:
                                                        # Insert new document
                                                        doc_id_data = {
                                                            "document_id":
                                                            document_id,
                                                            "document_name":
                                                            url,
                                                            "chatbot_id":
                                                            chatbot_id,
                                                            "created_at":
                                                            datetime.utcnow().
                                                            isoformat()
                                                        }
                                                        supabase.table(
                                                            'doc_ids').insert(
                                                                doc_id_data
                                                            ).execute()

                                                    # Check and update chatbot status if needed
                                                    credentials = supabase.table(
                                                        'credentials').select(
                                                            'chatbot_status'
                                                        ).eq(
                                                            'chatbot_id',
                                                            chatbot_id
                                                        ).execute()
                                                    if credentials.data and credentials.data[
                                                            0]['chatbot_status'] != 'Active':
                                                        supabase.table(
                                                            'credentials'
                                                        ).update({
                                                            'chatbot_status':
                                                            'Active'
                                                        }).eq(
                                                            'chatbot_id',
                                                            chatbot_id
                                                        ).execute()

                                                    processed_urls.append({
                                                        "url":
                                                        page_url,
                                                        "chatbot_id":
                                                        chatbot_id,
                                                        "status":
                                                        "success",
                                                        "document_id":
                                                        document_id
                                                    })

                                            except Exception as e:
                                                processed_urls.append({
                                                    "url":
                                                    page_url,
                                                    "chatbot_id":
                                                    chatbot_id,
                                                    "status":
                                                    "error",
                                                    "error":
                                                    str(e)
                                                })

                                # Clean up crawl file
                                os.remove(filename)

                    except Exception as e:
                        processed_urls.append({
                            "url": url,
                            "chatbot_id": chatbot_id,
                            "status": "error",
                            "error": str(e)
                        })

            except Exception as e:
                processed_urls.append({
                    "row_id": row.get('id'),
                    "status": "error",
                    "error": str(e)
                })

        return {"status": "success", "processed_urls": processed_urls}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing URLs: {str(e)}")


@app.get("/migrate_chats/")
async def migrate_chats():
    """Migrate chat messages from chat_messages to testing_zaps2 table"""
    try:
        # Get all unique user_id (chatbot_id) and session_id combinations
        unique_sessions = supabase.table('chat_messages').select(
            'user_id', 'session_id').execute()

        session_groups = {}
        # Group messages by user_id and session_id
        for row in unique_sessions.data:
            key = (row['user_id'], row['session_id'])
            if key not in session_groups:
                session_groups[key] = True

        processed_count = 0
        for user_id, session_id in session_groups:
            # Get all messages for this session ordered by timestamp
            messages = supabase.table('chat_messages')\
                .select('*')\
                .eq('user_id', user_id)\
                .eq('session_id', session_id)\
                .order('timestamp', desc=False)\
                .execute()

            if not messages.data:
                continue

            # Create array of message IDs
            message_ids = [msg['id'] for msg in messages.data]

            # Format messages for JSON storage with deduplication
            formatted_messages = []
            latest_timestamp = None
            seen_messages = set()  # Track unique message combinations

            for msg in messages.data:
                # Create unique key from role and content
                msg_key = f"{msg['message_type']}:{msg['message']}"

                if msg_key not in seen_messages:
                    seen_messages.add(msg_key)
                    formatted_messages.append({
                        "role":
                        "assistant"
                        if msg['message_type'] == 'assistant' else "user",
                        "content":
                        msg['message']
                    })
                    # Track latest timestamp for date_of_convo
                    if not latest_timestamp or msg[
                            'timestamp'] > latest_timestamp:
                        latest_timestamp = msg['timestamp']

            # Check if entry exists in testing_zaps2
            existing_entry = supabase.table('testing_zaps2')\
                .select('*')\
                .eq('chatbot_id', user_id)\
                .eq('session_id', session_id)\
                .execute()

            # Extract time from latest_timestamp
            convo_time = latest_timestamp.split('T')[1].split(
                '.')[0] if latest_timestamp else None

            if existing_entry.data:
                # Entry exists, update it
                existing_row = existing_entry.data[0]

                # Get existing sessions_added and filter out already processed IDs
                existing_ids = existing_row.get('sessions_added', []) or []
                new_ids = [id for id in message_ids if id not in existing_ids]

                if new_ids:  # Only update if there are new messages
                    # Update the entry
                    supabase.table('testing_zaps2').update({
                        'messages':
                        formatted_messages,
                        'sessions_added':
                        existing_ids + new_ids,
                        'date_of_convo':
                        latest_timestamp.split('T')[0]
                        if latest_timestamp else None,
                        'time_of_convo':
                        convo_time
                    }).eq('chatbot_id', user_id).eq('session_id',
                                                    session_id).execute()
                    processed_count += 1
            else:
                # Create new entry without re-logging to chat_messages
                supabase.table('testing_zaps2').insert({
                    'chatbot_id':
                    user_id,
                    'session_id':
                    session_id,
                    'messages':
                    formatted_messages,
                    'sessions_added':
                    message_ids,
                    'date_of_convo':
                    latest_timestamp.split('T')[0]
                    if latest_timestamp else None,
                    'time_of_convo':
                    convo_time,
                    'created_at':
                    datetime.utcnow().isoformat()
                }).execute()
                processed_count += 1

        return {
            "status": "success",
            "message":
            f"Successfully processed {processed_count} chat sessions",
            "total_unique_sessions": len(session_groups)
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Migration failed: {str(e)}")


@app.get("/delete_documents/")
async def delete_documents():
    """Delete orphaned document references and cleanup document records"""
    try:
        # Dictionary to store chatbot_ids and their valid document names
        chatbot_documents = {}
        deleted_docs = []

        # 1. Get all URLs from urls_uploaded table
        urls_result = supabase.table('urls_uploaded').select('*').execute()

        # Process URLs for each chatbot
        for row in urls_result.data:
            chatbot_id = row.get('chatbot_id')
            if not chatbot_id:
                continue

            # Initialize set for this chatbot if not exists
            if chatbot_id not in chatbot_documents:
                chatbot_documents[chatbot_id] = set()

            # Add URLs to the set
            url_links = row.get('url_links', {})
            if url_links and 'links' in url_links:
                for url in url_links['links']:
                    chatbot_documents[chatbot_id].add(url)

        # 2. Get documents from storage for each chatbot
        buckets = supabase.storage.from_('documentuploaded').list()
        for folder in buckets:
            chatbot_id = folder['name']

            # Initialize set for this chatbot if not exists
            if chatbot_id not in chatbot_documents:
                chatbot_documents[chatbot_id] = set()

            try:
                # List files in the folder
                files = supabase.storage.from_('documentuploaded').list(
                    chatbot_id)
                for file in files:
                    if file['name'].lower().endswith(('.pdf', '.docx')):
                        chatbot_documents[chatbot_id].add(file['name'])
            except Exception as e:
                print(
                    f"Error listing files for chatbot {chatbot_id}: {str(e)}")

        # 3. Process doc_ids table and cleanup
        for chatbot_id, valid_docs in chatbot_documents.items():
            # Get all doc_ids entries for this chatbot
            doc_ids_result = supabase.table('doc_ids').select('*').eq(
                'chatbot_id', chatbot_id).execute()

            for doc_entry in doc_ids_result.data:
                document_name = doc_entry.get('document_name')
                document_id = doc_entry.get('document_id')

                # If document_name not in valid_docs, delete entries
                if document_name not in valid_docs:
                    # Delete from doc_ids table
                    supabase.table('doc_ids').delete().eq(
                        'document_id', document_id).execute()

                    # Delete from documents table
                    supabase.table('documents').delete().eq(
                        'id', document_id).execute()

                    deleted_docs.append({
                        'chatbot_id': chatbot_id,
                        'document_name': document_name,
                        'document_id': document_id
                    })

        return {
            "status": "success",
            "message": f"Cleaned up {len(deleted_docs)} orphaned documents",
            "deleted_documents": deleted_docs,
            "processed_chatbots": len(chatbot_documents)
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to cleanup documents: {str(e)}")


@app.get("/update_old_chats/")
async def update_old_chats():
    """Update chat counts and limits based on testing_zaps2 table data with batch processing"""
    try:
        from datetime import datetime, timedelta
        import calendar
        import asyncio

        now = datetime.utcnow()
        month_start = now.replace(day=1).strftime('%Y-%m-%d')
        _, last_day = calendar.monthrange(now.year, now.month)
        month_end = now.replace(day=last_day).strftime('%Y-%m-%d')

        results = []
        seen_chatbots = set()
        start = 0
        batch_size = 800  # Process 800 rows at a time to stay within limits

        while True:
            # Fetch chatbots in batches using range
            chatbots = supabase.table("credentials")\
                .select("chatbot_id")\
                .range(start, start + batch_size - 1)\
                .execute()

            if not chatbots.data:
                break  # No more chatbots to process

            # Process chatbots in current batch
            batch_results = []
            for row in chatbots.data:
                chatbot_id = row.get('chatbot_id')
                if chatbot_id and chatbot_id not in seen_chatbots:
                    seen_chatbots.add(chatbot_id)

                # Count conversations for current month
                monthly_chats = supabase.table('testing_zaps2')\
                    .select('*')\
                    .eq('chatbot_id', chatbot_id)\
                    .gte('date_of_convo', month_start)\
                    .lte('date_of_convo', month_end)\
                    .execute()

                total_chats = len(
                    monthly_chats.data) if monthly_chats.data else 0

                # Get chatbot credentials
                credentials = supabase.table('credentials')\
                    .select('*')\
                    .eq('chatbot_id', chatbot_id)\
                    .execute()

                if credentials.data:
                    # Update total_chats and check limits for Basic plan
                    update_data = {'total_chats': total_chats}
                    plan = credentials.data[0].get('selected_plan')

                    if plan == 'Basic':
                        if total_chats >= 500:
                            update_data['chat_limit'] = 'Exceeded'
                            # Add to restricted_chatbots if not present
                            restricted = supabase.table('restricted_chatbots')\
                                .select('*')\
                                .eq('chatbot_id', chatbot_id)\
                                .execute()

                            if not restricted.data:
                                supabase.table('restricted_chatbots').insert({
                                    'chatbot_id':
                                    chatbot_id,
                                    'created_at':
                                    datetime.utcnow().isoformat()
                                }).execute()
                        else:
                            update_data['chat_limit'] = 'Within limit'
                            # Remove from restricted_chatbots if present
                            supabase.table('restricted_chatbots')\
                                .delete()\
                                .eq('chatbot_id', chatbot_id)\
                                .execute()

                    # Update credentials table
                    supabase.table('credentials')\
                        .update(update_data)\
                        .eq('chatbot_id', chatbot_id)\
                        .execute()

                    batch_results.append({
                        'chatbot_id':
                        chatbot_id,
                        'total_chats':
                        total_chats,
                        'plan':
                        plan,
                        'status':
                        update_data.get('chat_limit', 'N/A')
                    })

            # Add batch results to main results
            results.extend(batch_results)

            # Move to next batch
            start += batch_size

            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.5)

        return {
            'status': 'success',
            'message': f'Updated {len(results)} chatbots',
            'results': results
        }

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f'Failed to update chat counts: {str(e)}')


@app.get("/check_limits/")
async def check_limits():
    """Check chat limits for Basic plan users and update if exceeded"""
    try:
        # Get active chatbots with their plans
        active_chatbots = supabase.table('credentials')\
            .select('chatbot_id', 'selected_plan')\
            .eq('chatbot_status', 'Active')\
            .execute()

        if not active_chatbots.data:
            return {"status": "success", "message": "No active chatbots found"}

        results = []
        start = 0
        batch_size = 800  # Process 800 rows at a time

        while True:
            batch_results = []
            # Fetch chatbots in batches using range
            chatbots = supabase.table('credentials')\
                .select('chatbot_id', 'selected_plan')\
                .eq('chatbot_status', 'Active')\
                .range(start, start + batch_size - 1)\
                .execute()

            if not chatbots.data:
                break  # No more chatbots to process

            # Process each chatbot in the current batch
            for chatbot in chatbots.data:
                chatbot_id = chatbot['chatbot_id']
                selected_plan = chatbot['selected_plan']

                # Only process Basic plan chatbots
                if selected_plan == "Basic":
                    # Get current month's start and end dates
                    from datetime import datetime, timedelta
                    import calendar

                    now = datetime.utcnow()
                    month_start = now.replace(day=1,
                                              hour=0,
                                              minute=0,
                                              second=0,
                                              microsecond=0).isoformat()

                    # Get last day of current month
                    _, last_day = calendar.monthrange(now.year, now.month)
                    month_end = now.replace(day=last_day,
                                            hour=23,
                                            minute=59,
                                            second=59).isoformat()

                    # Get count of distinct session_ids for this chatbot for current month only
                    chat_count = supabase.table('chat_messages')\
                        .select('*', count='exact')\
                        .eq('user_id', chatbot_id)\
                        .gte('timestamp', month_start)\
                        .lte('timestamp', month_end)\
                        .execute()

                    total_chats = len(
                        set(msg['session_id'] for msg in chat_count.data if
                            msg.get('session_id'))) if chat_count.data else 0

                    # Update total_chats in credentials table
                    supabase.table('credentials')\
                        .update({'total_chats': total_chats})\
                        .eq('chatbot_id', chatbot_id)\
                        .execute()

                    # First set chat_limit to null to clear any existing value
                    supabase.table('credentials')\
                        .update({'chat_limit': None})\
                        .eq('chatbot_id', chatbot_id)\
                        .execute()

                    # Check if exceeded 500 chats/month limit
                    if total_chats > 2:
                        # Update chat_limit status in credentials table
                        supabase.table('credentials')\
                            .update({'chat_limit': 'Exceeded'})\
                            .eq('chatbot_id', chatbot_id)\
                            .execute()

                        # Add to restricted_chatbots if not already present
                        existing_restriction = supabase.table(
                            'restricted_chatbots')\
                            .select('*')\
                            .eq('chatbot_id', chatbot_id)\
                            .execute()

                        if not existing_restriction.data:
                            supabase.table('restricted_chatbots').insert({
                                'chatbot_id':
                                chatbot_id,
                                'created_at':
                                datetime.utcnow().isoformat()
                            }).execute()

                        batch_results.append({
                            "chatbot_id": chatbot_id,
                            "total_chats": total_chats,
                            "status": "Exceeded",
                            "plan": "Basic"
                        })
                    else:
                        batch_results.append({
                            "chatbot_id": chatbot_id,
                            "total_chats": total_chats,
                            "status": "Within limit",
                            "plan": "Basic"
                        })

            # Add batch results to main results
            results.extend(batch_results)

            # Move to next batch
            start += batch_size

            # Small delay between batches to avoid rate limits
            await asyncio.sleep(0.5)

        return {"status": "success", "checked_chatbots": results}

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Failed to check chat limits: {str(e)}")


def format_chatbot_response(response: str) -> str:
    """
    Formats the chatbot response to ensure consistent structure and formatting:

    1. Converts markdown-style bullet points to HTML <li> elements within a <ul> element.
    2. Preserves existing HTML formatting and line breaks.
    3. Sanitizes the output to prevent XSS vulnerabilities.
    """
    import re
    from bs4 import BeautifulSoup

    # Convert markdown bullet points to HTML
    response = re.sub(r'^(\s*\*\s+)(.+)$',
                      r'<li>\2</li>',
                      response,
                      flags=re.MULTILINE)
    if '<li>' in response:
        response = f"<ul>{response}</ul>"

    # Parse the response as HTML
    soup = BeautifulSoup(response, 'html.parser')

    # Sanitize HTML to prevent XSS and ensure valid structure
    sanitized_response = str(soup)

    return sanitized_response


def validate_response_formatting(response: str) -> tuple[str, list[str]]:
    """
    Enhanced 8-step comprehensive validation with advanced pattern detection for asterisk formatting.
    
    Fixes:
    1. Mixed asterisk patterns (text* → **text**)
    2. Single asterisks to proper bold format
    3. Trailing asterisks and orphaned symbols
    4. Standardized bullet points with enhanced detection
    5. Context-aware inline emphasis correction
    6. Advanced pattern normalization
    7. Structural formatting validation
    8. Final cleanup and consistency check

    Returns:
        A tuple containing the cleaned response and a list of formatting issues that were fixed.
    """
    import re

    formatting_issues = []
    cleaned_response = response
    original_response = response

    # Step 1: Advanced mixed asterisk pattern detection and normalization
    # Handle complex patterns like *text**, **text*, ***text***, etc.
    mixed_patterns = [
        # Pattern: *text** → **text**
        (r'^(\s*)(\*(?!\*))([^*\n]+?)(\*{2,})(\s*)$', r'\1**\3**\5'),
        # Pattern: **text* → **text**
        (r'^(\s*)(\*{2,})([^*\n]+?)(\*(?!\*))(\s*)$', r'\1**\3**\5'),
        # Pattern: ***text*** → **text**
        (r'^(\s*)(\*{3,})([^*\n]+?)(\*{3,})(\s*)$', r'\1**\3**\5'),
        # Pattern: *text* at line start → **text**
        (r'^(\s*)(\*(?!\*))([^*\n]+?)(\*(?!\*))(\s*)$', r'\1**\3**\5'),
    ]

    for pattern, replacement in mixed_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern,
                                  replacement,
                                  cleaned_response,
                                  flags=re.MULTILINE)
        if before != cleaned_response:
            formatting_issues.append(
                "Fixed mixed asterisk patterns at line boundaries")

    # Step 2: Context-aware single asterisk conversion with enhanced detection
    # Handle single asterisks that should be bold formatting
    single_asterisk_patterns = [
        # Single asterisk at start of line followed by colon (headers)
        (r'^(\s*)(\*(?!\*))([^*\n]*?:)(\s*)$', r'\1**\3**\4'),
        # Single asterisk at start of line (general headers)
        (r'^(\s*)(\*(?!\*))([^*\n]+?)(\s*)$', r'\1**\3**\4'),
        # Single asterisk mid-line emphasis
        (r'(\s)(\*(?!\*))([^*\s][^*\n]*?[^*\s])(\*(?!\*))(\s)', r'\1**\3**\5'),
    ]

    for pattern, replacement in single_asterisk_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern,
                                  replacement,
                                  cleaned_response,
                                  flags=re.MULTILINE)
        if before != cleaned_response:
            formatting_issues.append(
                "Converted single asterisks to proper bold format")

    # Step 3: Advanced trailing asterisk cleanup with pattern recognition
    trailing_patterns = [
        # Multiple trailing asterisks after bold text
        (r'(\*\*[^*\n]+?\*\*)(\*+)', r'\1'),
        # Trailing asterisks after normal text
        (r'([^*\n]+?)(\*+)(\s*)$', r'\1\3'),
        # Orphaned asterisks at line end
        (r'(\*{1})(\s*)$', r'\2'),
    ]

    for pattern, replacement in trailing_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern,
                                  replacement,
                                  cleaned_response,
                                  flags=re.MULTILINE)
        if before != cleaned_response:
            formatting_issues.append(
                "Cleaned trailing asterisks and orphaned symbols")

    # Step 4: Enhanced bullet point standardization with comprehensive detection
    bullet_patterns = [
        # Various bullet symbols to standardized bullet
        (r'^(\s*)([-\u2022\u2023\u25E6\u2043\u2219]|\*(?!\*))(\s+)(.+)$',
         r'\1• \4'),
        # Numbered lists that should be bullets
        (r'^(\s*)(\d+[\.\)])(\s+)(.+)$', r'\1• \4'),
        # Dash bullets with inconsistent spacing
        (r'^(\s*)(-)(\s*)(.+)$', r'\1• \4'),
    ]

    for pattern, replacement in bullet_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern,
                                  replacement,
                                  cleaned_response,
                                  flags=re.MULTILINE)
        if before != cleaned_response:
            formatting_issues.append(
                "Standardized bullet points with enhanced detection")

    # Step 5: Context-aware inline emphasis correction with advanced logic
    # Handle inline emphasis while preserving intentional formatting
    inline_patterns = [
        # Single asterisk emphasis within sentences (not at boundaries)
        (r'(?<=\w)(\*(?!\*))([^*\n]{1,50}?)(\*(?!\*))(?=\w)', r'**\2**'),
        # Single asterisk around important terms
        (r'(\s)(\*(?!\*))([A-Z][^*\n]*?)(\*(?!\*))(\s|[.,!?])', r'\1**\3**\5'),
        # Fix broken bold formatting
        (r'(\*{2,})([^*\n]+?)(\*{1})(?!\*)', r'**\2**'),
    ]

    for pattern, replacement in inline_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern, replacement, cleaned_response)
        if before != cleaned_response:
            formatting_issues.append(
                "Applied context-aware inline emphasis correction")

    # Step 6: Advanced pattern normalization for consistency
    normalization_patterns = [
        # Normalize multiple bold markers
        (r'\*{4,}([^*\n]+?)\*{4,}', r'**\1**'),
        # Fix asymmetric bold markers
        (r'\*{3}([^*\n]+?)\*{2}', r'**\1**'),
        (r'\*{2}([^*\n]+?)\*{3}', r'**\1**'),
        # Remove space around bold content
        (r'\*\*(\s+)([^*\n]+?)(\s+)\*\*', r'**\2**'),
    ]

    for pattern, replacement in normalization_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern, replacement, cleaned_response)
        if before != cleaned_response:
            formatting_issues.append("Applied advanced pattern normalization")

    # Step 7: Structural formatting validation and enhancement
    # Ensure proper spacing and structure
    structural_patterns = [
        # Add proper spacing after bold headers
        (r'(\*\*[^*\n]+?\*\*)(\n)([^\n•])', r'\1\2\n\3'),
        # Ensure bullet points have consistent indentation
        (r'^(•)([^\s])', r'\1 \2'),
        # Fix spacing around bullet points
        (r'(\n)(•\s+[^\n]+)(\n)([^\n•])', r'\1\2\3\n\4'),
    ]

    for pattern, replacement in structural_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern,
                                  replacement,
                                  cleaned_response,
                                  flags=re.MULTILINE)
        if before != cleaned_response:
            formatting_issues.append(
                "Enhanced structural formatting validation")

    # Step 8: Final cleanup and consistency check
    final_cleanup_patterns = [
        # Remove any remaining single asterisks not part of bold formatting
        (r'(?<!\*)(\*)(?!\*)(?![^*\n]*\*)', r''),
        # Ensure no empty bold tags
        (r'\*\*\s*\*\*', r''),
        # Clean up excessive whitespace
        (r'\n{3,}', r'\n\n'),
        # Ensure proper line ending format
        (r'(\*\*[^*\n]+?\*\*)\s*(\n)', r'\1\2'),
    ]

    for pattern, replacement in final_cleanup_patterns:
        before = cleaned_response
        cleaned_response = re.sub(pattern, replacement, cleaned_response)
        if before != cleaned_response:
            formatting_issues.append(
                "Applied final cleanup and consistency check")

    # Comprehensive validation check
    if cleaned_response != original_response:
        # Verify no malformed patterns remain
        malformed_check = re.search(r'(?<!\*)\*(?!\*)|(?<!\*)\*{3,}(?!\*)',
                                    cleaned_response)
        if malformed_check:
            formatting_issues.append(
                "Warning: Some malformed patterns may remain")

    return cleaned_response, formatting_issues


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)