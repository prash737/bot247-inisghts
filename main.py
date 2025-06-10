
from supabase import create_client, Client
import time
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import io
import logging
from typing import List, Dict, Optional
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analytics.log')
    ]
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing required environment variables: SUPABASE_URL and SUPABASE_KEY")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set as environment variables")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    raise

MAX_RETRIES = 3
BATCH_SIZE = 1000
PLOT_TIMEOUT = 30


def retry_on_failure(max_retries: int = MAX_RETRIES):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


@retry_on_failure()
def fetch_all_conversations(chatbot_id: str) -> List[Dict]:
    try:
        count_query = supabase.table('testing_zaps2').select(
            '*', count='exact').eq('chatbot_id', chatbot_id)
        count_response = count_query.execute()
        total_count = count_response.count
        
        if total_count == 0:
            logger.info(f"No conversations found for chatbot_id: {chatbot_id}")
            return []

        logger.info(f"Fetching {total_count} conversations for chatbot_id: {chatbot_id}")
        
        all_conversations = []
        for offset in range(0, total_count, BATCH_SIZE):
            try:
                batch_query = supabase.table('testing_zaps2') \
                    .select('messages, date_of_convo, created_at, id, chatbot_id') \
                    .eq('chatbot_id', chatbot_id) \
                    .range(offset, offset + BATCH_SIZE - 1)
                batch_response = batch_query.execute()
                
                if batch_response.data:
                    all_conversations.extend(batch_response.data)
                    logger.debug(f"Fetched batch {offset//BATCH_SIZE + 1}/{(total_count//BATCH_SIZE) + 1}")
                else:
                    logger.warning(f"Empty batch at offset {offset}")
                    
            except Exception as e:
                logger.error(f"Error fetching batch at offset {offset}: {e}")
                continue
                
        logger.info(f"Successfully fetched {len(all_conversations)} conversations")
        return all_conversations
        
    except Exception as e:
        logger.error(f"Error fetching conversations for chatbot {chatbot_id}: {e}")
        raise


@retry_on_failure()
def get_distinct_chatbot_ids() -> List[str]:
    try:
        response = supabase.rpc('get_distinct_chatbot_ids').execute()
        
        if response.data:
            chatbot_ids = [item['chatbot_id'] for item in response.data if item.get('chatbot_id')]
            logger.info(f"Found {len(chatbot_ids)} distinct chatbot IDs")
            return chatbot_ids
            
    except Exception as e:
        logger.warning(f"RPC call failed, falling back to manual method: {e}")
        
    try:
        all_ids = set()
        offset = 0
        while True:
            response = supabase.table('testing_zaps2').select(
                'chatbot_id').range(offset, offset + BATCH_SIZE - 1).execute()
            if not response.data:
                break

            batch_ids = set(item['chatbot_id'] for item in response.data
                            if item.get('chatbot_id'))
            all_ids.update(batch_ids)

            if len(response.data) < BATCH_SIZE:
                break

            offset += BATCH_SIZE

        result = list(all_ids)
        logger.info(f"Found {len(result)} distinct chatbot IDs using fallback method")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching distinct chatbot IDs: {e}")
        return []


def validate_conversation_data(conversations: List[Dict]) -> List[Dict]:
    valid_conversations = []
    
    for convo in conversations:
        try:
            if not isinstance(convo.get('messages'), list):
                continue
                
            if not convo.get('id'):
                continue
                
            if convo.get('date_of_convo'):
                try:
                    datetime.strptime(convo['date_of_convo'], "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid date format for conversation {convo.get('id')}")
                    continue
            
            valid_conversations.append(convo)
            
        except Exception as e:
            logger.warning(f"Error validating conversation {convo.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Validated {len(valid_conversations)}/{len(conversations)} conversations")
    return valid_conversations


def find_unanswered_queries(conversations: List[Dict]) -> List[str]:
    unanswered_queries = []
    unanswered_patterns = ["Oops", "I don't know", "I'm not sure", "I can't help"]

    for convo in conversations:
        try:
            if "messages" not in convo or not isinstance(convo["messages"], list):
                continue

            messages = convo["messages"]
            for i in range(1, len(messages)):
                try:
                    current_msg = messages[i]
                    if (current_msg.get("role") == "assistant" and
                        any(pattern in current_msg.get("content", "") for pattern in unanswered_patterns)):
                        
                        if i > 0 and messages[i - 1].get("role") == "user":
                            user_content = messages[i - 1].get("content", "").strip()
                            if user_content:
                                unanswered_queries.append(user_content)
                                
                except (IndexError, KeyError, TypeError) as e:
                    logger.debug(f"Error processing message in conversation {convo.get('id')}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error processing conversation {convo.get('id', 'unknown')}: {e}")
            continue

    return unanswered_queries


def save_plot_to_supabase(plt_fig, plot_name: str, chatbot_id: str, period: int) -> bool:
    PLOT_BUCKET_NAME = "plots"
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = f"{chatbot_id}/{today}/{period}"

    try:
        if not plot_name or not chatbot_id:
            logger.error("Invalid plot_name or chatbot_id")
            return False
            
        buf = io.BytesIO()
        plt_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        buf.seek(0)
        file_content = buf.read()
        buf.close()

        if len(file_content) == 0:
            logger.error(f"Empty plot generated for {plot_name}")
            return False

        filename = f"{dir_path}/{plot_name}.png"
        response = supabase.storage.from_(PLOT_BUCKET_NAME).upload(
            filename, file_content, {
                "contentType": "image/png",
                "upsert": "true"
            })
        
        if response:
            logger.debug(f"Successfully uploaded {plot_name}")
            return True
        else:
            logger.error(f"Failed to upload {plot_name}")
            return False

    except Exception as e:
        logger.error(f"Error saving plot {plot_name}: {e}")
        return False


def safe_plot_generation(func):
    def wrapper(*args, **kwargs):
        try:
            plt.ioff()
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in plot generation {func.__name__}: {e}")
            return False
        finally:
            plt.close('all')
            if plt.get_fignums():
                for fignum in plt.get_fignums():
                    plt.close(fignum)
    return wrapper


@safe_plot_generation
def generate_message_distribution(user_queries: List[str], assistant_responses: List[str], 
                                chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        total_queries = len(user_queries) if user_queries else 0
        total_responses = len(assistant_responses) if assistant_responses else 0
        
        if total_queries == 0 and total_responses == 0:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            labels = ['User Queries', 'Assistant Responses']
            sizes = [total_queries, total_responses]
            colors = ['#3498DB', '#E74C3C']
            
            non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
            
            if non_zero_data:
                labels, sizes, colors = zip(*non_zero_data)
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                  autopct='%1.1f%%', startangle=90,
                                                  wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3),
                                                  textprops={'fontsize': 14, 'fontweight': 'bold'})
                
                total_messages = sum(sizes)
                ax.text(0, 0.1, f'{total_messages:,}', ha='center', va='center', 
                        fontsize=20, fontweight='bold', color='#2C3E50')
                ax.text(0, -0.1, 'Total Messages', ha='center', va='center', 
                        fontsize=12, color='#7F8C8D')
        
        ax.set_title('Message Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Message distribution plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating message distribution: {e}")
        return False


@safe_plot_generation
def generate_message_length_analysis(user_queries: List[str], assistant_responses: List[str], 
                                   chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not user_queries and not assistant_responses:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            query_lengths = [len(str(q).split()) for q in user_queries if q and str(q).strip()]
            response_lengths = [len(str(r).split()) for r in assistant_responses if r and str(r).strip()]
            
            if not query_lengths and not response_lengths:
                ax.text(0.5, 0.5, 'No valid message data available', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
            else:
                max_length = max(
                    max(query_lengths) if query_lengths else 0, 
                    max(response_lengths) if response_lengths else 0
                )
                
                if max_length > 0:
                    bins = np.linspace(0, min(max_length, 100), 25)
                    
                    if query_lengths:
                        ax.hist(query_lengths, bins=bins, alpha=0.7, label='User Queries', 
                                color='#3498DB', density=True, edgecolor='white')
                    if response_lengths:
                        ax.hist(response_lengths, bins=bins, alpha=0.7, label='Assistant Responses', 
                                color='#E74C3C', density=True, edgecolor='white')
                    
                    if query_lengths:
                        avg_query = np.mean(query_lengths)
                        ax.axvline(avg_query, color='#2980B9', linestyle='--', linewidth=2,
                                   label=f'Avg Query: {avg_query:.1f} words')
                    
                    if response_lengths:
                        avg_response = np.mean(response_lengths)
                        ax.axvline(avg_response, color='#C0392B', linestyle='--', linewidth=2,
                                   label=f'Avg Response: {avg_response:.1f} words')
                    
                    ax.set_xlabel('Message Length (words)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
                    ax.grid(True, alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
        
        ax.set_title('Message Length Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Message length analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating message length analysis: {e}")
        return False


@safe_plot_generation
def generate_message_complexity_analysis(user_queries: List[str], assistant_responses: List[str], 
                                       chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not user_queries and not assistant_responses:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            all_messages = [msg for msg in (user_queries + assistant_responses) if msg and str(msg).strip()]
            
            if not all_messages:
                ax.text(0.5, 0.5, 'No valid messages available', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
            else:
                complexity_data = {'Simple\n(1-5 words)': 0, 'Moderate\n(6-15 words)': 0, 
                                 'Complex\n(16-30 words)': 0, 'Very Complex\n(31+ words)': 0}
                
                for msg in all_messages:
                    try:
                        word_count = len(str(msg).split())
                        if word_count <= 5:
                            complexity_data['Simple\n(1-5 words)'] += 1
                        elif word_count <= 15:
                            complexity_data['Moderate\n(6-15 words)'] += 1
                        elif word_count <= 30:
                            complexity_data['Complex\n(16-30 words)'] += 1
                        else:
                            complexity_data['Very Complex\n(31+ words)'] += 1
                    except Exception as e:
                        logger.debug(f"Error processing message complexity: {e}")
                        continue
                
                categories = list(complexity_data.keys())
                values = list(complexity_data.values())
                colors_complexity = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
                
                if sum(values) > 0:
                    bars = ax.bar(categories, values, color=colors_complexity, alpha=0.8, 
                                  edgecolor='white', linewidth=2)
                    ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    max_value = max(values) if values else 0
                    for bar, value in zip(bars, values):
                        if value > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, 
                                   bar.get_height() + max_value*0.02,
                                   f'{value:,}', ha='center', va='bottom', 
                                   fontweight='bold', fontsize=11)
        
        ax.set_title('Message Complexity Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Message complexity analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating message complexity analysis: {e}")
        return False


@safe_plot_generation
def generate_key_performance_metrics(user_queries: List[str], assistant_responses: List[str], 
                                   chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        if not user_queries and not assistant_responses:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
        else:
            total_queries = len(user_queries) if user_queries else 0
            total_responses = len(assistant_responses) if assistant_responses else 0
            all_messages = [msg for msg in (user_queries + assistant_responses) if msg and str(msg).strip()]
            
            try:
                valid_queries = [q for q in user_queries if q and str(q).strip()]
                valid_responses = [r for r in assistant_responses if r and str(r).strip()]
                
                avg_query_length = np.mean([len(str(q).split()) for q in valid_queries]) if valid_queries else 0
                avg_response_length = np.mean([len(str(r).split()) for r in valid_responses]) if valid_responses else 0
                response_ratio = total_responses / total_queries if total_queries > 0 else 0
                verbosity_index = avg_response_length / avg_query_length if avg_query_length > 0 else 0
                
                metrics = [
                    ('Total Conversations', f'{len(set(all_messages)):,}'),
                    ('Response Rate', f'{response_ratio:.2f}:1'),
                    ('Avg Query Length', f'{avg_query_length:.1f} words'),
                    ('Avg Response Length', f'{avg_response_length:.1f} words'),
                    ('Verbosity Index', f'{verbosity_index:.2f}x'),
                    ('Total Word Count', f'{sum(len(str(msg).split()) for msg in all_messages):,}')
                ]
                
                y_start = 0.85
                for i, (metric, value) in enumerate(metrics):
                    y_pos = y_start - (i * 0.12)
                    
                    ax.text(0.05, y_pos, metric, fontsize=16, fontweight='bold', 
                            transform=ax.transAxes, va='center')
                    ax.text(0.95, y_pos, value, fontsize=16, transform=ax.transAxes, 
                            va='center', ha='right', color='#2C3E50',
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECF0F1", 
                                     edgecolor='#BDC3C7', linewidth=1))
                
                if total_queries > 0:
                    performance_score = min(100, (response_ratio * 50) + (min(avg_response_length, 20) * 2.5))
                    performance_level = 'Excellent' if performance_score >= 80 else 'Good' if performance_score >= 60 else 'Fair'
                    performance_color = '#27AE60' if performance_score >= 80 else '#F39C12' if performance_score >= 60 else '#E74C3C'
                    
                    ax.text(0.5, 0.15, f'Performance: {performance_level} ({performance_score:.0f}/100)', 
                            transform=ax.transAxes, ha='center', va='center', fontsize=18, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor=performance_color, alpha=0.2, 
                                     edgecolor=performance_color, linewidth=2))
                     
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
                ax.text(0.5, 0.5, 'Error calculating metrics', 
                       ha='center', va='center', fontsize=16, color='#E74C3C')
        
        ax.set_title('Key Performance Metrics', fontsize=24, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Key performance metrics plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating key performance metrics: {e}")
        return False


@safe_plot_generation
def generate_sentiment_analysis(user_queries: List[str], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not user_queries:
            ax.text(0.5, 0.5, 'No queries available for sentiment analysis',
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
            valid_queries = 0
            
            for query in user_queries:
                try:
                    if not query or not str(query).strip():
                        continue
                        
                    analysis = TextBlob(str(query))
                    polarity = analysis.sentiment.polarity
                    
                    if polarity > 0.1:
                        sentiments["Positive"] += 1
                    elif polarity < -0.1:
                        sentiments["Negative"] += 1
                    else:
                        sentiments["Neutral"] += 1
                    
                    valid_queries += 1
                    
                except Exception as e:
                    logger.debug(f"Error analyzing sentiment for query: {e}")
                    continue
            
            if valid_queries == 0:
                ax.text(0.5, 0.5, 'No valid queries for sentiment analysis',
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
            else:
                ordered_labels = ["Positive", "Neutral", "Negative"]
                ordered_values = [sentiments[label] for label in ordered_labels]
                colors = ["#27AE60", "#F39C12", "#E74C3C"]
                
                non_zero_data = [(label, value, color) for label, value, color in 
                               zip(ordered_labels, ordered_values, colors) if value > 0]
                
                if non_zero_data:
                    labels, values, colors = zip(*non_zero_data)
                    
                    wedges, texts, autotexts = ax.pie(
                        values, labels=labels, colors=colors,
                        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values))})',
                        startangle=90, pctdistance=0.75, labeldistance=1.15,
                        textprops={'fontsize': 14, 'fontweight': 'bold'},
                        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3)
                    )
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                    
                    total_queries = sum(values)
                    ax.text(0, 0.15, 'SENTIMENT', ha='center', va='center', 
                            fontsize=16, fontweight='bold', color='#2C3E50')
                    ax.text(0, 0, f'{total_queries:,}', ha='center', va='center', 
                            fontsize=24, fontweight='bold', color='#2C3E50')
                    ax.text(0, -0.15, 'QUERIES', ha='center', va='center', 
                            fontsize=12, fontweight='bold', color='#7F8C8D')
        
        ax.set_title('Sentiment Analysis', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating sentiment analysis: {e}")
        return False


@safe_plot_generation
def generate_sentiment_score_distribution(user_queries: List[str], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not user_queries:
            ax.text(0.5, 0.5, 'No queries available for sentiment analysis',
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            sentiment_scores = []
            
            for query in user_queries:
                try:
                    if not query or not str(query).strip():
                        continue
                        
                    analysis = TextBlob(str(query))
                    sentiment_scores.append(analysis.sentiment.polarity)
                    
                except Exception as e:
                    logger.debug(f"Error analyzing sentiment score: {e}")
                    continue
            
            if not sentiment_scores:
                ax.text(0.5, 0.5, 'No valid sentiment scores available',
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
            else:
                ax.hist(sentiment_scores, bins=30, color='#3498DB', alpha=0.7, 
                       edgecolor='white', density=True)
                ax.axvline(0, color='#34495E', linestyle='-', linewidth=2, alpha=0.7, 
                          label='Neutral Line')
                
                mean_sentiment = np.mean(sentiment_scores)
                ax.axvline(mean_sentiment, color='#E74C3C', linestyle='--', linewidth=2, 
                          label=f'Average: {mean_sentiment:.3f}')
                
                ax.set_xlabel('Sentiment Score', fontsize=14, fontweight='bold')
                ax.set_ylabel('Density', fontsize=14, fontweight='bold')
                ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                y_max = ax.get_ylim()[1]
                ax.text(-0.8, y_max*0.8, 'Very\nNegative', ha='center', va='center', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#E74C3C", alpha=0.3))
                ax.text(0, y_max*0.9, 'Neutral', ha='center', va='center', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F39C12", alpha=0.3))
                ax.text(0.8, y_max*0.8, 'Very\nPositive', ha='center', va='center', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#27AE60", alpha=0.3))
        
        ax.set_title('Sentiment Score Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Sentiment score distribution plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating sentiment score distribution: {e}")
        return False


@safe_plot_generation
def generate_chat_volume_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if period == 0:
            monthly_counts = {}
            for convo in valid_conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    month_key = convo_date.strftime("%Y-%m")
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                except Exception as e:
                    logger.debug(f"Error processing date: {e}")
                    continue

            if not monthly_counts:
                ax.text(0.5, 0.5, 'No conversation data available', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                dates = sorted(monthly_counts.keys())
                counts = [monthly_counts[date] for date in dates]
                formatted_dates = [datetime.strptime(date, "%Y-%m").strftime("%b %Y") for date in dates]

                ax.plot(formatted_dates, counts, marker='o', linewidth=3, markersize=8, 
                        color='#3498DB', markerfacecolor='#E74C3C', markeredgecolor='white', markeredgewidth=2)
                ax.fill_between(formatted_dates, counts, alpha=0.3, color='#3498DB')
                
                ax.set_ylabel('Number of Conversations', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if counts:
                    max_count = max(counts)
                    max_idx = counts.index(max_count)
                    ax.annotate(f'Peak: {max_count:,}', xy=(max_idx, max_count), 
                               xytext=(max_idx, max_count + max_count*0.1),
                               arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
                               fontsize=12, ha='center', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="#E74C3C", alpha=0.3))
                
        else:
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_counts = {}

            for i in range(period):
                date_key = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_counts[date_key] = 0

            for convo in valid_conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    date_key = convo["date_of_convo"]
                    if date_key in daily_counts:
                        daily_counts[date_key] += 1
                except Exception as e:
                    logger.debug(f"Error processing daily data: {e}")
                    continue

            dates = sorted(daily_counts.keys())
            counts = [daily_counts[date] for date in dates]
            formatted_dates = [datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d") for date in dates]

            if sum(counts) == 0:
                ax.text(0.5, 0.5, f'No conversations in the last {period} days', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.plot(formatted_dates, counts, marker='o', linewidth=3, markersize=6, 
                        color='#3498DB', markerfacecolor='#E74C3C', markeredgecolor='white', markeredgewidth=2)
                ax.fill_between(formatted_dates, counts, alpha=0.3, color='#3498DB')
                
                ax.set_ylabel('Number of Conversations', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=11)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if len(counts) > 2:
                    try:
                        z = np.polyfit(range(len(counts)), counts, 1)
                        p = np.poly1d(z)
                        ax.plot(formatted_dates, p(range(len(counts))), "--", color='#E74C3C', 
                                linewidth=2, alpha=0.8, label=f'Trend: {"↗" if z[0] > 0 else "↘"}')
                        ax.legend(fontsize=12)
                    except np.RankWarning:
                        logger.debug("Could not fit trend line")
        
        title = f'Chat Volume Trend (Last {period} Days)' if period > 0 else 'Chat Volume Trend (All Time)'
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Chat volume plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating chat volume plot: {e}")
        return False


@safe_plot_generation
def generate_peak_hours_activity_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        hourly_activity = {hour: 0 for hour in range(24)}
        
        valid_conversations = validate_conversation_data(conversations)
        
        for convo in valid_conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo and convo["created_at"]:
                    timestamp_str = convo["created_at"]
                    if timestamp_str.endswith('Z'):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    hour = timestamp.hour
                    hourly_activity[hour] += 1
            except Exception as e:
                logger.debug(f"Error processing timestamp: {e}")
                continue

        total_activity = sum(hourly_activity.values())
        
        if total_activity == 0:
            ax.text(0.5, 0.5, 'No activity data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            hours = list(range(24))
            activity_counts = [hourly_activity[hour] for hour in hours]
            
            bars = ax.bar(hours, activity_counts, color='#3498DB', alpha=0.7, 
                         edgecolor='white', linewidth=1)
            
            if max(activity_counts) > 0:
                peak_hour = hours[activity_counts.index(max(activity_counts))]
                bars[peak_hour].set_color('#E74C3C')
                bars[peak_hour].set_alpha(0.9)
            
            ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Hourly Activity Pattern {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Peak hours activity plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating peak hours activity plot: {e}")
        return False


@safe_plot_generation
def generate_day_of_week_activity_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        daily_activity = {day: 0 for day in range(7)}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        valid_conversations = validate_conversation_data(conversations)
        
        for convo in valid_conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo and convo["created_at"]:
                    timestamp_str = convo["created_at"]
                    if timestamp_str.endswith('Z'):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    day_of_week = timestamp.weekday()
                    daily_activity[day_of_week] += 1
                elif "date_of_convo" in convo and convo["date_of_convo"]:
                    date_obj = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    day_of_week = date_obj.weekday()
                    daily_activity[day_of_week] += 1
            except Exception as e:
                logger.debug(f"Error processing day of week: {e}")
                continue

        total_activity = sum(daily_activity.values())
        
        if total_activity == 0:
            ax.text(0.5, 0.5, 'No activity data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            day_counts = [daily_activity[day] for day in range(7)]
            day_bars = ax.bar(day_names, day_counts, color='#27AE60', alpha=0.7, 
                             edgecolor='white', linewidth=1)
            
            if max(day_counts) > 0:
                busiest_day_idx = day_counts.index(max(day_counts))
                day_bars[busiest_day_idx].set_color('#E74C3C')
                day_bars[busiest_day_idx].set_alpha(0.9)
            
            ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            max_count = max(day_counts) if day_counts else 0
            for bar, count in zip(day_bars, day_counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max_count*0.02,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Day of Week Activity {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Day of week activity plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating day of week activity plot: {e}")
        return False


@safe_plot_generation
def generate_business_hours_analysis_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        hourly_activity = {hour: 0 for hour in range(24)}
        
        valid_conversations = validate_conversation_data(conversations)
        
        for convo in valid_conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo and convo["created_at"]:
                    timestamp_str = convo["created_at"]
                    if timestamp_str.endswith('Z'):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    hour = timestamp.hour
                    hourly_activity[hour] += 1
            except Exception as e:
                logger.debug(f"Error processing business hours timestamp: {e}")
                continue

        total_activity = sum(hourly_activity.values())
        
        if total_activity == 0:
            ax.text(0.5, 0.5, 'No activity data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            business_hours = list(range(9, 17))
            after_hours = list(range(0, 9)) + list(range(17, 24))
            
            business_activity = sum(hourly_activity[hour] for hour in business_hours)
            after_hours_activity = sum(hourly_activity[hour] for hour in after_hours)
            
            time_periods = ['Business Hours\n(9AM-5PM)', 'After Hours\n(5PM-9AM)']
            time_counts = [business_activity, after_hours_activity]
            time_colors = ['#3498DB', '#E67E22']
            
            non_zero_data = [(period, count, color) for period, count, color in 
                           zip(time_periods, time_counts, time_colors) if count > 0]
            
            if non_zero_data:
                periods, counts, colors = zip(*non_zero_data)
                
                wedges, texts, autotexts = ax.pie(counts, labels=periods, colors=colors,
                                                  autopct='%1.1f%%', startangle=90,
                                                  textprops={'fontsize': 12, 'fontweight': 'bold'},
                                                  wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
                
                total_count = sum(counts)
                ax.text(0, 0, f'{total_count:,}\nTotal', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='#2C3E50')
            else:
                ax.text(0.5, 0.5, 'No activity data available', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Business vs After Hours {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Business hours analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating business hours analysis plot: {e}")
        return False


@safe_plot_generation
def generate_conversation_quality_analysis(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            quality_scores = []
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                    
                    if not user_messages:
                        continue

                    total_messages = len(messages)
                    user_count = len(user_messages)
                    assistant_count = len(assistant_messages)
                    response_ratio = assistant_count / user_count if user_count > 0 else 0
                    
                    unanswered_patterns = ["Oops", "I don't know", "I'm not sure", "I can't help"]
                    has_unanswered = any(
                        any(pattern in msg.get("content", "") for pattern in unanswered_patterns)
                        for msg in assistant_messages
                    )
                    
                    quality_score = min(100, (
                        (min(total_messages, 20) / 20 * 30) +
                        (min(response_ratio, 2) / 2 * 25) +
                        (30 if not has_unanswered else 0) +
                        (15 if total_messages > 5 else total_messages * 3)
                    ))
                    
                    quality_scores.append(quality_score)
                    
                except Exception as e:
                    logger.debug(f"Error calculating quality score: {e}")
                    continue
            
            if not quality_scores:
                ax.text(0.5, 0.5, 'No valid conversation data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                score_ranges = ['Excellent\n(80-100)', 'Good\n(60-79)', 'Fair\n(40-59)', 'Poor\n(0-39)']
                score_counts = [
                    sum(1 for score in quality_scores if score >= 80),
                    sum(1 for score in quality_scores if 60 <= score < 80),
                    sum(1 for score in quality_scores if 40 <= score < 60),
                    sum(1 for score in quality_scores if score < 40)
                ]
                score_colors = ['#27AE60', '#3498DB', '#F39C12', '#E74C3C']
                
                bars = ax.bar(score_ranges, score_counts, color=score_colors, alpha=0.8, 
                               edgecolor='white', linewidth=2)
                ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                total_convos = len(quality_scores)
                max_count = max(score_counts) if score_counts else 0
                for bar, count in zip(bars, score_counts):
                    if count > 0:
                        percentage = (count / total_convos) * 100
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + max_count*0.02,
                                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                                fontweight='bold', fontsize=10)
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Quality Score Distribution {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Conversation quality analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating conversation quality analysis: {e}")
        return False


@safe_plot_generation
def generate_quality_correlation_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            quality_scores = []
            message_depths = []
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                    
                    if not user_messages:
                        continue

                    total_messages = len(messages)
                    user_count = len(user_messages)
                    assistant_count = len(assistant_messages)
                    response_ratio = assistant_count / user_count if user_count > 0 else 0
                    
                    unanswered_patterns = ["Oops", "I don't know", "I'm not sure", "I can't help"]
                    has_unanswered = any(
                        any(pattern in msg.get("content", "") for pattern in unanswered_patterns)
                        for msg in assistant_messages
                    )
                    
                    quality_score = min(100, (
                        (min(total_messages, 20) / 20 * 30) +
                        (min(response_ratio, 2) / 2 * 25) +
                        (30 if not has_unanswered else 0) +
                        (15 if total_messages > 5 else total_messages * 3)
                    ))
                    
                    quality_scores.append(quality_score)
                    message_depths.append(total_messages)
                    
                except Exception as e:
                    logger.debug(f"Error calculating correlation data: {e}")
                    continue
            
            if not quality_scores or len(quality_scores) < 2:
                ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                scatter = ax.scatter(message_depths, quality_scores, alpha=0.6, s=60, 
                           c=quality_scores, cmap='RdYlGn', edgecolors='white', linewidth=1)
                
                plt.colorbar(scatter, ax=ax, label='Quality Score')
                
                if len(message_depths) > 2:
                    try:
                        z = np.polyfit(message_depths, quality_scores, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(message_depths), max(message_depths), 100)
                        ax.plot(x_trend, p(x_trend), "--", color='#E74C3C', linewidth=2, alpha=0.8)
                    except (np.RankWarning, np.linalg.LinAlgError):
                        logger.debug("Could not fit trend line for correlation")
                
                ax.set_xlabel('Total Messages per Conversation', fontsize=12, fontweight='bold')
                ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                try:
                    correlation = np.corrcoef(message_depths, quality_scores)[0, 1]
                    if not np.isnan(correlation):
                        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", alpha=0.8),
                                fontsize=11, fontweight='bold')
                except Exception as e:
                    logger.debug(f"Could not calculate correlation: {e}")
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Message Depth vs Quality Correlation {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Quality correlation plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating quality correlation plot: {e}")
        return False


@safe_plot_generation
def generate_resolution_analysis_plot(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            resolution_status = []
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                    
                    if not user_messages:
                        continue

                    unanswered_patterns = ["Oops", "I don't know", "I'm not sure", "I can't help"]
                    has_unanswered = any(
                        any(pattern in msg.get("content", "") for pattern in unanswered_patterns)
                        for msg in assistant_messages
                    )
                    resolution_status.append("Resolved" if not has_unanswered else "Unresolved")
                    
                except Exception as e:
                    logger.debug(f"Error analyzing resolution status: {e}")
                    continue
            
            if not resolution_status:
                ax.text(0.5, 0.5, 'No valid conversation data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                resolved_count = resolution_status.count("Resolved")
                unresolved_count = resolution_status.count("Unresolved")
                
                if resolved_count + unresolved_count > 0:
                    resolution_data = [resolved_count, unresolved_count]
                    resolution_labels = ['Resolved', 'Unresolved']
                    resolution_colors = ['#27AE60', '#E74C3C']
                    
                    non_zero_data = [(label, count, color) for label, count, color in 
                                   zip(resolution_labels, resolution_data, resolution_colors) if count > 0]
                    
                    if non_zero_data:
                        labels, counts, colors = zip(*non_zero_data)
                        
                        wedges, texts, autotexts = ax.pie(counts, labels=labels, 
                                                          colors=colors, autopct='%1.1f%%',
                                                          startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                                          wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
                        
                        resolution_rate = (resolved_count / (resolved_count + unresolved_count)) * 100
                        ax.text(0, 0, f'{resolution_rate:.1f}%\nResolution\nRate', ha='center', va='center', 
                                fontsize=14, fontweight='bold', color='#2C3E50')
                    else:
                        ax.text(0.5, 0.5, 'No resolution data available', 
                               ha='center', va='center', fontsize=16, color='#7F8C8D')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Problem Resolution Analysis {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Resolution analysis plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating resolution analysis plot: {e}")
        return False


@safe_plot_generation
def generate_user_engagement_funnel(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No engagement data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            engagement_stages = {
                "Initial Contact": 0,
                "Engaged (2-4 msgs)": 0,
                "Active (5-9 msgs)": 0,
                "Highly Engaged (10-19 msgs)": 0,
                "Power Users (20+ msgs)": 0
            }
            
            user_message_counts = []
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    user_count = len(user_messages)
                    
                    if user_count == 0:
                        continue
                    
                    user_message_counts.append(user_count)
                    
                    if user_count >= 1:
                        engagement_stages["Initial Contact"] += 1
                    if 2 <= user_count <= 4:
                        engagement_stages["Engaged (2-4 msgs)"] += 1
                    elif 5 <= user_count <= 9:
                        engagement_stages["Active (5-9 msgs)"] += 1
                    elif 10 <= user_count <= 19:
                        engagement_stages["Highly Engaged (10-19 msgs)"] += 1
                    elif user_count >= 20:
                        engagement_stages["Power Users (20+ msgs)"] += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing engagement data: {e}")
                    continue
            
            if not user_message_counts:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stages = list(engagement_stages.keys())
                values = list(engagement_stages.values())
                colors = ['#1ABC9C', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C']
                
                y_positions = np.arange(len(stages))[::-1]
                bars = ax.barh(y_positions, values, color=colors, alpha=0.8, 
                              edgecolor='white', linewidth=2)
                
                initial_users = values[0] if values else 1
                max_value = max(values) if values else 0
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        percentage = (value / initial_users) * 100
                        ax.text(value + max_value * 0.02, bar.get_y() + bar.get_height()/2,
                                f'{value:,} ({percentage:.1f}%)', va='center', fontweight='bold', fontsize=11)
                
                ax.set_yticks(y_positions)
                ax.set_yticklabels(stages, fontsize=11)
                ax.set_xlabel('Number of Users', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'User Engagement Funnel {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "User engagement funnel plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating user engagement funnel: {e}")
        return False


@safe_plot_generation
def generate_user_message_distribution(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No engagement data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            user_message_counts = []
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    user_count = len(user_messages)
                    
                    if user_count > 0:
                        user_message_counts.append(user_count)
                        
                except Exception as e:
                    logger.debug(f"Error processing user message count: {e}")
                    continue
            
            if not user_message_counts:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                max_messages = max(user_message_counts)
                bins = min(20, max_messages) if max_messages > 0 else 10
                
                ax.hist(user_message_counts, bins=bins, 
                        color='#3498DB', alpha=0.7, edgecolor='white', linewidth=1)
                
                avg_messages = np.mean(user_message_counts)
                median_messages = np.median(user_message_counts)
                
                ax.axvline(avg_messages, color='#E74C3C', linestyle='--', linewidth=2,
                           label=f'Average: {avg_messages:.1f}')
                ax.axvline(median_messages, color='#F39C12', linestyle='--', linewidth=2,
                           label=f'Median: {median_messages:.1f}')
                
                ax.set_xlabel('Messages per User', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
                ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'User Message Distribution {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "User message distribution plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating user message distribution: {e}")
        return False


@safe_plot_generation
def generate_engagement_level_distribution(conversations: List[Dict], chatbot_id: str, period: int):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        valid_conversations = validate_conversation_data(conversations)
        
        if not valid_conversations:
            ax.text(0.5, 0.5, 'No engagement data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            engagement_stages = {
                "Initial Contact": 0,
                "Engaged (2-4 msgs)": 0,
                "Active (5-9 msgs)": 0,
                "Highly Engaged (10-19 msgs)": 0,
                "Power Users (20+ msgs)": 0
            }
            
            for convo in valid_conversations:
                if "messages" not in convo or not isinstance(convo["messages"], list):
                    continue

                try:
                    messages = convo["messages"]
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    user_count = len(user_messages)
                    
                    if user_count == 0:
                        continue
                    
                    if user_count >= 1:
                        engagement_stages["Initial Contact"] += 1
                    if 2 <= user_count <= 4:
                        engagement_stages["Engaged (2-4 msgs)"] += 1
                    elif 5 <= user_count <= 9:
                        engagement_stages["Active (5-9 msgs)"] += 1
                    elif 10 <= user_count <= 19:
                        engagement_stages["Highly Engaged (10-19 msgs)"] += 1
                    elif user_count >= 20:
                        engagement_stages["Power Users (20+ msgs)"] += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing engagement level: {e}")
                    continue
            
            non_zero_engagement = [(stage, value) for stage, value in engagement_stages.items() if value > 0]
            
            if non_zero_engagement:
                stages = list(engagement_stages.keys())
                colors = ['#1ABC9C', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C']
                
                pie_labels, pie_values = zip(*non_zero_engagement)
                pie_colors = [colors[stages.index(label)] for label in pie_labels]
                
                wedges, texts, autotexts = ax.pie(pie_values, labels=pie_labels, colors=pie_colors,
                                                  autopct='%1.1f%%', startangle=90,
                                                  textprops={'fontsize': 10, 'fontweight': 'bold'},
                                                  wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))
                
                total_users = sum(pie_values)
                ax.text(0, 0, f'{total_users:,}\nUsers', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='#2C3E50')
            else:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Engagement Level Distribution {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        
        success = save_plot_to_supabase(plt, "Engagement level distribution plot", chatbot_id, period)
        return success
        
    except Exception as e:
        logger.error(f"Error generating engagement level distribution: {e}")
        return False


def get_conversation_insights(chatbot_id: str, period: int) -> Optional[Dict]:
    try:
        logger.info(f"Generating insights for chatbot {chatbot_id}, period: {period} days")
        
        processed_ids = set()
        all_conversations = fetch_all_conversations(chatbot_id)
        
        if not all_conversations:
            logger.warning(f"No conversations found for chatbot {chatbot_id}")
            return None

        all_conversations = validate_conversation_data(all_conversations)
        
        filtered_conversations = []
        if period > 0:
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = current_date - timedelta(days=period)

            for convo in all_conversations:
                try:
                    if "date_of_convo" not in convo or not convo["date_of_convo"]:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    if convo_date >= cutoff_date:
                        filtered_conversations.append(convo)
                        processed_ids.add(convo["id"])
                except Exception as e:
                    logger.debug(f"Error processing date for conversation {convo.get('id')}: {e}")
                    continue
        else:
            filtered_conversations = all_conversations
            processed_ids = set(convo["id"] for convo in all_conversations if convo.get("id"))

        conversations = filtered_conversations
        logger.info(f"Processing {len(conversations)} conversations for insights")

        user_queries = []
        assistant_responses = []

        for convo in conversations:
            if "messages" not in convo or not isinstance(convo["messages"], list):
                continue

            try:
                messages = convo["messages"]
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                        
                    role = message.get("role")
                    content = message.get("content", "")
                    
                    if role == "user" and content.strip():
                        user_queries.append(content.strip())
                    elif role == "assistant" and content.strip():
                        assistant_responses.append(content.strip())
                        
            except Exception as e:
                logger.debug(f"Error processing messages for conversation {convo.get('id')}: {e}")
                continue

        logger.info(f"Extracted {len(user_queries)} user queries and {len(assistant_responses)} assistant responses")

        unanswered_queries = find_unanswered_queries(conversations)
        unanswered_query_counts = dict(Counter(unanswered_queries).most_common())

        unanswered_queries_json = {
            "queries": unanswered_query_counts,
            "total_count": len(unanswered_queries)
        }

        top_user_queries_dict = dict(Counter(user_queries).most_common(10))
        top_user_queries_json = {"queries": top_user_queries_dict}

        plot_results = {}
        
        logger.info("Generating message analysis plots...")
        plot_results['message_distribution'] = generate_message_distribution(user_queries, assistant_responses, chatbot_id, period)
        plot_results['message_length'] = generate_message_length_analysis(user_queries, assistant_responses, chatbot_id, period)
        plot_results['message_complexity'] = generate_message_complexity_analysis(user_queries, assistant_responses, chatbot_id, period)
        plot_results['performance_metrics'] = generate_key_performance_metrics(user_queries, assistant_responses, chatbot_id, period)
        
        logger.info("Generating sentiment analysis plots...")
        plot_results['sentiment_analysis'] = generate_sentiment_analysis(user_queries, chatbot_id, period)
        plot_results['sentiment_scores'] = generate_sentiment_score_distribution(user_queries, chatbot_id, period)
        
        logger.info("Generating chat volume plot...")
        plot_results['chat_volume'] = generate_chat_volume_plot(conversations, chatbot_id, period)
        
        logger.info("Generating activity pattern plots...")
        plot_results['peak_hours'] = generate_peak_hours_activity_plot(conversations, chatbot_id, period)
        plot_results['day_of_week'] = generate_day_of_week_activity_plot(conversations, chatbot_id, period)
        plot_results['business_hours'] = generate_business_hours_analysis_plot(conversations, chatbot_id, period)
        
        logger.info("Generating quality analysis plots...")
        plot_results['quality_analysis'] = generate_conversation_quality_analysis(conversations, chatbot_id, period)
        plot_results['quality_correlation'] = generate_quality_correlation_plot(conversations, chatbot_id, period)
        plot_results['resolution_analysis'] = generate_resolution_analysis_plot(conversations, chatbot_id, period)
        
        logger.info("Generating engagement analysis plots...")
        plot_results['engagement_funnel'] = generate_user_engagement_funnel(conversations, chatbot_id, period)
        plot_results['message_distribution_users'] = generate_user_message_distribution(conversations, chatbot_id, period)
        plot_results['engagement_levels'] = generate_engagement_level_distribution(conversations, chatbot_id, period)

        successful_plots = sum(1 for result in plot_results.values() if result)
        logger.info(f"Successfully generated {successful_plots}/{len(plot_results)} plots")

        current_date = datetime.now().date().isoformat()
        insights_data = {
            "chatbot_id": chatbot_id,
            "total_conversations": len(conversations),
            "total_user_queries": len(user_queries),
            "total_assistant_responses": len(assistant_responses),
            "date_of_convo": current_date,
            "period_range": period,
            "unanswered_queries": unanswered_queries_json,
            "top_user_queries": top_user_queries_json,
            "created_at": datetime.now().isoformat(),
            "plot_generation_success": plot_results
        }

        try:
            existing_record = supabase.table('insights') \
                .select('id') \
                .eq('chatbot_id', chatbot_id) \
                .eq('period_range', period) \
                .eq('date_of_convo', current_date) \
                .execute()

            if existing_record.data:
                record_id = existing_record.data[0]['id']
                supabase.table('insights') \
                    .update(insights_data) \
                    .eq('id', record_id) \
                    .execute()
                logger.info(f"Updated existing insights record for chatbot {chatbot_id}, period {period}")
            else:
                supabase.table('insights').insert(insights_data).execute()
                logger.info(f"Inserted new insights record for chatbot {chatbot_id}, period {period}")

        except Exception as e:
            logger.error(f"Error saving insights to database: {e}")

        return insights_data
        
    except Exception as e:
        logger.error(f"Error generating conversation insights for chatbot {chatbot_id}: {e}")
        return None


@retry_on_failure()
def update_tokens():
    try:
        logger.info("Starting token update process")
        chatbot_ids = get_distinct_chatbot_ids()
        today = datetime.now().date().isoformat()
        
        if not chatbot_ids:
            logger.warning("No chatbot IDs found for token update")
            return

        successful_updates = 0
        
        for chatbot_id in chatbot_ids:
            try:
                response = supabase.table('testing_zaps2') \
                    .select('input_tokens, output_tokens') \
                    .eq('chatbot_id', chatbot_id) \
                    .eq('date_of_convo', today) \
                    .execute()

                if not response.data:
                    logger.debug(f"No conversations found for chatbot {chatbot_id} today")
                    continue

                total_input_tokens = 0
                total_output_tokens = 0
                
                for row in response.data:
                    try:
                        input_tokens = row.get('input_tokens')
                        output_tokens = row.get('output_tokens')
                        
                        if input_tokens is not None:
                            total_input_tokens += int(input_tokens)
                        if output_tokens is not None:
                            total_output_tokens += int(output_tokens)
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error converting token values for chatbot {chatbot_id}: {e}")
                        continue

                if total_input_tokens == 0 and total_output_tokens == 0:
                    logger.debug(f"No token usage for chatbot {chatbot_id} today")
                    continue

                existing_tokens = supabase.table('chat_tokens') \
                    .select('input_tokens, output_tokens') \
                    .eq('chatbot_id', chatbot_id) \
                    .execute()

                if existing_tokens.data:
                    try:
                        current_input = int(existing_tokens.data[0].get('input_tokens') or 0)
                        current_output = int(existing_tokens.data[0].get('output_tokens') or 0)
                        
                        supabase.table('chat_tokens') \
                            .update({
                                'input_tokens': current_input + total_input_tokens,
                                'output_tokens': current_output + total_output_tokens
                            }) \
                            .eq('chatbot_id', chatbot_id) \
                            .execute()
                            
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error updating existing tokens for chatbot {chatbot_id}: {e}")
                        continue
                else:
                    supabase.table('chat_tokens') \
                        .insert({
                            'chatbot_id': chatbot_id,
                            'input_tokens': total_input_tokens,
                            'output_tokens': total_output_tokens
                        }) \
                        .execute()

                successful_updates += 1
                logger.debug(f"Updated tokens for chatbot {chatbot_id}")

            except Exception as e:
                logger.error(f"Error updating tokens for chatbot {chatbot_id}: {e}")
                continue

        logger.info(f"Token update process completed. Updated {successful_updates}/{len(chatbot_ids)} chatbots")
        
    except Exception as e:
        logger.error(f"Error in update_tokens: {e}")
        raise


def process_chatbot_data():
    try:
        logger.info(f"Starting data processing at {datetime.now()}")
        chatbot_ids = get_distinct_chatbot_ids()
        
        if not chatbot_ids:
            logger.warning("No chatbot IDs found for processing")
            return
            
        logger.info(f"Found {len(chatbot_ids)} distinct chatbot IDs")

        periods = [0, 2, 7, 10]
        total_tasks = len(chatbot_ids) * len(periods)
        completed_tasks = 0
        successful_tasks = 0

        for chatbot_id in chatbot_ids:
            try:
                logger.info(f"Processing chatbot ID: {chatbot_id}")

                for period in periods:
                    try:
                        logger.info(f"Analyzing {period}-day period for {chatbot_id}")
                        insights = get_conversation_insights(chatbot_id, period)
                        
                        if insights:
                            logger.info(f"Successfully generated insights for {chatbot_id} ({period} days)")
                            logger.info(f"Total conversations: {insights['total_conversations']}")
                            logger.info(f"Total queries: {insights['total_user_queries']}")
                            successful_tasks += 1
                        else:
                            logger.warning(f"Failed to generate insights for {chatbot_id} ({period} days)")
                            
                    except Exception as e:
                        logger.error(f"Error processing period {period} for chatbot {chatbot_id}: {e}")
                    finally:
                        completed_tasks += 1
                        
                    progress = (completed_tasks / total_tasks) * 100
                    logger.info(f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%)")

            except Exception as e:
                logger.error(f"Error processing chatbot ID {chatbot_id}: {e}")
                continue

        logger.info(f"Completed data processing at {datetime.now()}")
        logger.info(f"Success rate: {successful_tasks}/{total_tasks} ({(successful_tasks/total_tasks)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error in process_chatbot_data: {e}")
        raise


@retry_on_failure()
def update_job_status(job_statuses: List[Dict]):
    try:
        current_time = datetime.now().isoformat()

        status_data = {
            "created_at": current_time,
            "job_status": job_statuses
        }

        supabase.table('insights_schedule').insert(status_data).execute()
        logger.info("Job statuses updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating job status: {e}")
        raise


if __name__ == "__main__":
    plt.ioff()
    
    job_statuses = []
    overall_success = True
    
    try:
        logger.info("Starting analytics job execution")
        
        try:
            logger.info("Starting token update job")
            update_tokens()
            job_statuses.append({
                "job_name": "update_tokens",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "error": None
            })
            logger.info("Token update job completed successfully")
            
        except Exception as e:
            logger.error(f"Token update job failed: {e}")
            job_statuses.append({
                "job_name": "update_tokens", 
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            overall_success = False

        try:
            logger.info("Starting data processing job")
            process_chatbot_data()
            job_statuses.append({
                "job_name": "process_chatbot_data",
                "status": "success", 
                "timestamp": datetime.now().isoformat(),
                "error": None
            })
            logger.info("Data processing job completed successfully")
            
        except Exception as e:
            logger.error(f"Data processing job failed: {e}")
            job_statuses.append({
                "job_name": "process_chatbot_data",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            overall_success = False

        try:
            update_job_status(job_statuses)
        except Exception as e:
            logger.error(f"Failed to update job statuses: {e}")

        if overall_success:
            logger.info("All analytics jobs completed successfully")
        else:
            logger.warning("Some analytics jobs failed. Check logs for details.")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        job_statuses.append({
            "job_name": "main_execution",
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })
        
        try:
            update_job_status(job_statuses)
        except Exception as status_error:
            logger.error(f"Failed to update failure status: {status_error}")
            
        sys.exit(1)
    
    finally:
        plt.close('all')
        logger.info("Analytics execution completed")
