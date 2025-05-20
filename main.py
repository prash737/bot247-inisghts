
from supabase import create_client, Client
import time
from datetime import datetime

# Initialize Supabase client
SUPABASE_URL = "https://zsivtypgrrcttzhtfjsf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpzaXZ0eXBncnJjdHR6aHRmanNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzgzMzU5NTUsImV4cCI6MjA1MzkxMTk1NX0.3cAMZ4LPTqgIc8z6D8LRkbZvEhP_ffI3Wka0-QDSIys"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_distinct_chatbot_ids():
    try:
        response = supabase.table('testing_zaps2').select('chatbot_id').execute()
        distinct_ids = set(item['chatbot_id'] for item in response.data if item.get('chatbot_id'))
        return list(distinct_ids)
    except Exception as e:
        print(f"Error fetching distinct chatbot IDs: {e}")
        return []

def process_chatbot_data():
    print(f"\nStarting data processing at {datetime.now()}")
    chatbot_ids = get_distinct_chatbot_ids()
    print(f"Found {len(chatbot_ids)} distinct chatbot IDs")
    
    for chatbot_id in chatbot_ids:
        try:
            print(f"\nProcessing chatbot ID: {chatbot_id}")
            
            # Get conversation insights
            insights = get_conversation_insights(chatbot_id)
            if insights:
                print(f"Successfully generated insights for {chatbot_id}")
            
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
