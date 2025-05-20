from supabase import create_client, Client
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import io
import re

# Initialize Supabase client

PLOT_BUCKET_NAME = "plots"

SUPABASE_URL = "https://zsivtypgrrcttzhtfjsf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpzaXZ0eXBncnJjdHR6aHRmanNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzgzMzU5NTUsImV4cCI6MjA1MzkxMTk1NX0.3cAMZ4LPTqgIc8z6D8LRkbZvEhP_ffI3Wka0-QDSIys"
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
        generate_monthly_conversation_heatmap(conversations, chatbot_id)

        insights = {
            "total_conversations": len(conversations),
            "total_user_queries": len(user_queries),
            "total_assistant_responses": len(assistant_responses),
            "total_unanswered_queries": len(unanswered_queries),
            "top_10_queries": Counter(user_queries).most_common(10),
            "top_10_unanswered_queries": Counter(unanswered_queries).most_common(10),
            "plots": {
                "top_queries": None,
                "message_distribution": None,
                "sentiment_analysis": None,
                "monthly_heatmap": None
            }
        }

        return insights

    except Exception as e:
        print(f"Error reading insights: {str(e)}")
        return None

def save_plot_to_supabase(plot, filename, chatbot_id, period):
    today = datetime.now().strftime("%Y-%m-%d")
    dir_name = f"{chatbot_id}/{today}/{period}"

    # Only delete existing files when saving the first plot
    if filename == "Top 10 user queries":
        try:
            existing_files = supabase.storage.from_(PLOT_BUCKET_NAME).list(path=dir_name)
            for file in existing_files:
                try:
                    supabase.storage.from_(PLOT_BUCKET_NAME).remove([f"{dir_name}/{file['name']}"])
                    print(f"Deleted: {file['name']}")
                except Exception as e:
                    print(f"Error deleting {file['name']}: {e}")
        except Exception as e:
            print(f"Error listing files: {e}")

    # Save new plot
    filename_with_timestamp = f"{dir_name}/{filename}.png"

    buf = io.BytesIO()
    plot.savefig(buf, format="png")
    buf.seek(0)
    file_content = buf.read()
    buf.close()

    try:
        response = supabase.storage.from_(PLOT_BUCKET_NAME).upload(
            filename_with_timestamp, file_content, {"content-type": "image/png"})
        if not response:
            print("Error uploading plots to Supabase.")
    except Exception as e:
        print(f"Error saving plot to Supabase: {e}")

def save_plot_to_memory():
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

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
        sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for query in user_queries:
            analysis = TextBlob(query)
            if analysis.sentiment.polarity > 0:
                sentiments["Positive"] += 1
            elif analysis.sentiment.polarity < 0:
                sentiments["Negative"] += 1
            else:
                sentiments["Neutral"] += 1

        plt.figure(figsize=(10, 10), dpi=150)
        plt.pie(sentiments.values(), labels=sentiments.keys(), 
                autopct="%1.1f%%", colors=["#90EE90", "#FFB366", "#FF7F7F"])
        plt.title("Sentiment Analysis of User Queries", pad=20, fontsize=24)
        save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
    except Exception as e:
        print(f"Error generating sentiment analysis: {e}")
    finally:
        plt.close()

def generate_monthly_conversation_heatmap(conversations, chatbot_id):
    try:
        conversation_counts = {}
        for convo in conversations:
            try:
                date_of_convo = datetime.strptime(convo['date_of_convo'], "%Y-%m-%d")
                month = date_of_convo.month
                day = date_of_convo.day
                if month not in conversation_counts:
                    conversation_counts[month] = {}
                if day not in conversation_counts[month]:
                    conversation_counts[month][day] = 0
                conversation_counts[month][day] += 1
            except Exception as e:
                print(f"Error processing conversation date: {e}")
                continue

        heatmap_matrix = np.zeros((12, 31))
        for month in conversation_counts:
            for day in conversation_counts[month]:
                heatmap_matrix[month - 1, day - 1] = conversation_counts[month][day]

        plt.figure(figsize=(15, 8), dpi=150)
        plt.imshow(heatmap_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='NUMBER OF CONVERSATIONS')
        plt.xlabel('DAY OF MONTH')
        plt.ylabel('MONTH')
        plt.yticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.xticks(range(0, 31, 5), [str(i + 1) for i in range(0, 31, 5)])
        plt.title('Monthly Conversation Distribution', fontsize=24, pad=20)
        save_plot_to_supabase(plt, "Monthly conversation heatmap", chatbot_id, "all")
    except Exception as e:
        print(f"Error generating heatmap: {e}")
    finally:
        plt.close()

def main():
    # Example usage
    chatbot_id = "sibm_1"
    period = 7
    
    # Get conversation insights
    print("\nGetting conversation insights...")
    insights = get_conversation_insights(chatbot_id, period)
    if insights:
        print(f"\nTotal Conversations: {insights['total_conversations']}")
        print(f"Total User Queries: {insights['total_user_queries']}")
        print(f"Total Assistant Responses: {insights['total_assistant_responses']}")
        print(f"Total Unanswered Queries: {insights['total_unanswered_queries']}")
        
        print("\nTop 10 Queries:")
        for query, count in insights['top_10_queries']:
            print(f"- '{query}' (Count: {count})")
            
        print("\nTop 10 Unanswered Queries:")
        for query, count in insights['top_10_unanswered_queries']:
            print(f"- '{query}' (Count: {count})")
    
    # Extract leads
    print("\nExtracting leads...")
    conversations = fetch_all_conversations(chatbot_id)
    leads = extract_leads(conversations)
    print(f"\nTotal leads found: {leads['total_leads']}")
    for lead in leads['leads']:
        print(f"\nLead from conversation {lead['conversation_id']}:")
        print(f"Date: {lead['date']}")
        print(f"Name: {lead['found_data']['name']}")
        print(f"Email: {lead['found_data']['email']}")
        print(f"Phone: {lead['found_data']['phone']}")

if __name__ == "__main__":
    main()