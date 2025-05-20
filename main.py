
from supabase import create_client, Client
import time
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re
import io

# Initialize Supabase client
import os
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all_conversations(chatbot_id):
    count_query = supabase.table('testing_zaps2').select('*', count='exact').eq('chatbot_id', chatbot_id)
    count_response = count_query.execute()
    total_count = count_response.count

    all_conversations = []
    for offset in range(0, total_count, 1000):
        batch_query = supabase.table('testing_zaps2') \
            .select('messages, date_of_convo, created_at, id, chatbot_id') \
            .eq('chatbot_id', chatbot_id) \
            .range(offset, offset + 999)
        batch_response = batch_query.execute()
        all_conversations.extend(batch_response.data)
    return all_conversations

def get_distinct_chatbot_ids():
    try:
        response = supabase.table('testing_zaps2').select('chatbot_id').execute()
        distinct_ids = set(item['chatbot_id'] for item in response.data if item.get('chatbot_id'))
        return list(distinct_ids)
    except Exception as e:
        print(f"Error fetching distinct chatbot IDs: {e}")
        return []

def find_unanswered_queries(conversations):
    unanswered_queries = []
    unanswered_message = "Oops"

    for convo in conversations:
        if "messages" not in convo:
            continue

        messages = convo["messages"]
        for i in range(1, len(messages)):
            if (messages[i].get("role") == "assistant" and 
                unanswered_message in messages[i].get("content", "")):
                if i > 0 and messages[i - 1].get("role") == "user":
                    unanswered_queries.append(messages[i - 1].get("content", ""))

    return unanswered_queries

def save_plot_to_supabase(plt, plot_name, chatbot_id, period):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    return buffer

def generate_top_10_user_queries(user_queries, chatbot_id, period):
    try:
        if not user_queries:
            plt.figure(figsize=(15, 10))
            plt.text(0.5, 0.5, 'No queries available', horizontalalignment='center', fontsize=20)
            plt.axis('off')
        else:
            query_labels, query_values = zip(*Counter(user_queries).most_common(10))
            plt.figure(figsize=(12, 8), dpi=150)
            plt.barh(query_labels, query_values, color="skyblue")
            plt.xlabel("FREQUENCY", fontsize=20)
            plt.title("Top 10 User Queries", pad=20, fontsize=24)
        save_plot_to_supabase(plt, "Top 10 user queries", chatbot_id, period)
    except Exception as e:
        print(f"Error generating top queries plot: {e}")
    finally:
        plt.close()

def generate_message_distribution(user_queries, assistant_responses, chatbot_id, period):
    try:
        labels = ["USER QUERIES", "ASSISTANT RESPONSES"]
        sizes = [len(user_queries), len(assistant_responses)]
        if not sizes or sum(sizes) == 0:
            sizes = [1, 1]
        plt.figure(figsize=(10, 10), dpi=150)
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#ff9999", "#66b3ff"])
        save_plot_to_supabase(plt, "Message distribution plot", chatbot_id, period)
    except Exception as e:
        print(f"Error generating message distribution: {e}")
    finally:
        plt.close()

def generate_sentiment_analysis(user_queries, chatbot_id, period):
    try:
        if not user_queries:
            plt.figure(figsize=(10, 10), dpi=150)
            plt.text(0.5, 0.5, 'No queries available for sentiment analysis', 
                    horizontalalignment='center', fontsize=20)
            plt.axis('off')
            save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
            return

        sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for query in user_queries:
            analysis = TextBlob(query)
            if analysis.sentiment.polarity > 0:
                sentiments["Positive"] += 1
            elif analysis.sentiment.polarity < 0:
                sentiments["Negative"] += 1
            else:
                sentiments["Neutral"] += 1

        # Only create pie chart if we have data
        if sum(sentiments.values()) > 0:
            plt.figure(figsize=(10, 10), dpi=150)
            plt.pie(list(sentiments.values()), labels=list(sentiments.keys()), 
                    autopct="%1.1f%%", colors=["#90EE90", "#FFB366", "#FF7F7F"])
            plt.title("Sentiment Analysis of User Queries", pad=20, fontsize=24)
        else:
            plt.figure(figsize=(10, 10), dpi=150)
            plt.text(0.5, 0.5, 'No sentiment data available', 
                    horizontalalignment='center', fontsize=20)
            plt.axis('off')
            
        save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
    except Exception as e:
        print(f"Error generating sentiment analysis: {e}")
    finally:
        plt.close()

def get_conversation_insights(chatbot_id, period=7):
    try:
        processed_ids = set()
        all_conversations = fetch_all_conversations(chatbot_id)
        
        filtered_conversations = []
        if period > 0:
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = current_date - timedelta(days=period)

            for convo in all_conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    if convo_date >= cutoff_date:
                        filtered_conversations.append(convo)
                        processed_ids.add(convo["id"])
                except Exception as e:
                    print(f"Error processing date for conversation: {e}")
                    continue
        else:
            filtered_conversations = all_conversations
            processed_ids = set(convo["id"] for convo in all_conversations)

        conversations = filtered_conversations
        user_queries = []
        assistant_responses = []

        for convo in conversations:
            if "messages" not in convo:
                continue
            
            messages = convo["messages"]
            for message in messages:
                if message.get("role") == "user":
                    user_queries.append(message.get("content", ""))
                elif message.get("role") == "assistant":
                    assistant_responses.append(message.get("content", ""))

        unanswered_queries = find_unanswered_queries(conversations)
        
        # Generate all visualizations
        generate_top_10_user_queries(user_queries, chatbot_id, period)
        generate_message_distribution(user_queries, assistant_responses, chatbot_id, period)
        generate_sentiment_analysis(user_queries, chatbot_id, period)

        insights = {
            "total_conversations": len(conversations),
            "total_user_queries": len(user_queries),
            "total_assistant_responses": len(assistant_responses),
            "total_unanswered_queries": len(unanswered_queries),
            "top_10_queries": Counter(user_queries).most_common(10),
            "top_10_unanswered_queries": Counter(unanswered_queries).most_common(10),
        }

        return insights

    except Exception as e:
        print(f"Error reading insights: {str(e)}")
        return None

def extract_leads(conversations):
    leads = {"total_leads": 0, "leads": []}
    
    chatbot_conversations = {}
    for convo in conversations:
        chatbot_id = convo.get("chatbot_id", "unknown")
        if chatbot_id not in chatbot_conversations:
            chatbot_conversations[chatbot_id] = []
        chatbot_conversations[chatbot_id].append(convo)

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(?:\+?(?:91|1)?[-\s]?)?(?:\(?\d{3}\)?[-\s]?)?\d{3}[-\s]?\d{4}'
    name_prefixes = ['my name is', 'i am', 'this is', 'i\'m', 'call me']
    MIN_PHONE_DIGITS = 8

    for chatbot_id, chatbot_convos in chatbot_conversations.items():
        chatbot_leads = []

        for convo in chatbot_convos:
            if "messages" not in convo:
                continue

            convo_lead = {
                "conversation_id": convo.get("id", "unknown"),
                "date": convo.get("date_of_convo", ""),
                "found_data": {"name": None, "email": None, "phone": None}
            }

            has_valid_email = False
            has_valid_phone = False

            for message in convo["messages"]:
                if message.get("role") != "user":
                    continue

                content = message.get("content", "")
                
                # Extract email
                emails = re.findall(email_pattern, content)
                if emails and not convo_lead["found_data"]["email"]:
                    convo_lead["found_data"]["email"] = emails[0]
                    has_valid_email = True

                # Extract phone
                phones = re.findall(phone_pattern, content)
                if phones and not convo_lead["found_data"]["phone"]:
                    phone = re.sub(r'[^0-9+]', '', phones[0])
                    if len(phone.replace('+', '')) >= MIN_PHONE_DIGITS:
                        convo_lead["found_data"]["phone"] = phone
                        has_valid_phone = True

                # Extract name
                if not convo_lead["found_data"]["name"]:
                    content_lower = content.lower()
                    for prefix in name_prefixes:
                        if prefix in content_lower:
                            name_part = content_lower.split(prefix, 1)[1].strip()
                            name_match = re.search(r'([^.,;!?\n]*)', name_part)
                            if name_match:
                                potential_name = name_match.group(1).strip()
                                stopwords = [' and ', ' from ', ' i ', ' am ', ' a ', ' an ', ' the ', ' here ', ' to ', ' for ']
                                for word in stopwords:
                                    if f" {word} " in f" {potential_name} ":
                                        potential_name = potential_name.split(word)[0].strip()
                                potential_name = ' '.join(word.capitalize() for word in potential_name.split())
                                if 1 <= len(potential_name.split()) <= 3 and 2 < len(potential_name) < 40:
                                    convo_lead["found_data"]["name"] = potential_name
                                    break

            if has_valid_email or has_valid_phone:
                chatbot_leads.append(convo_lead)
                leads["leads"].append(convo_lead)
                leads["total_leads"] += 1

        try:
            for lead in chatbot_leads:
                lead_data = {
                    "chatbot_id": chatbot_id,
                    "name": lead["found_data"]["name"] or "000",
                    "email": lead["found_data"]["email"] or "000",
                    "phone": lead["found_data"]["phone"] or "000"
                }

                if lead_data["email"] != "000" or lead_data["phone"] != "000":
                    existing_lead = None
                    if lead_data["email"] != "000":
                        existing_lead = supabase.table('collected_leads').select('*').eq('email', lead_data["email"]).execute()

                    if not existing_lead or not existing_lead.data:
                        if lead_data["phone"] != "000":
                            existing_lead = supabase.table('collected_leads').select('*').eq('phone', lead_data["phone"]).execute()

                    if not existing_lead or not existing_lead.data:
                        supabase.table('collected_leads').insert(lead_data).execute()
                        print(f"Inserted lead for chatbot: {chatbot_id}, name: {lead_data['name']}, email: {lead_data['email']}, phone: {lead_data['phone']}")
        except Exception as e:
            print(f"Error inserting leads for chatbot {chatbot_id}: {e}")

    return leads

def process_chatbot_data():
    print(f"\nStarting data processing at {datetime.now()}")
    chatbot_ids = get_distinct_chatbot_ids()
    print(f"Found {len(chatbot_ids)} distinct chatbot IDs")
    
    periods = [0, 2, 7, 10]  # Analysis periods in days
    
    for chatbot_id in chatbot_ids:
        try:
            print(f"\nProcessing chatbot ID: {chatbot_id}")
            
            # Get conversation insights for different periods
            for period in periods:
                print(f"\nAnalyzing {period}-day period:")
                insights = get_conversation_insights(chatbot_id, period)
                if insights:
                    print(f"Successfully generated insights for {chatbot_id} ({period} days)")
                    print(f"Total conversations: {insights['total_conversations']}")
                    print(f"Total queries: {insights['total_user_queries']}")
            
            # Extract leads
            conversations = fetch_all_conversations(chatbot_id)
            leads = extract_leads(conversations)
            print(f"Processed {leads['total_leads']} leads for {chatbot_id}")
            
        except Exception as e:
            print(f"Error processing chatbot ID {chatbot_id}: {e}")
            continue
    
    print(f"\nCompleted data processing at {datetime.now()}")

if __name__ == "__main__":
    process_chatbot_data()
