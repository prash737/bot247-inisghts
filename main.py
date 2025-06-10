# Updated get_conversation_insights function to store insights in the "insights" table with specified JSON formats and date.
from supabase import create_client, Client
import time
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import re
import io
import calendar

# Initialize Supabase client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_conversations(chatbot_id):
    count_query = supabase.table('testing_zaps2').select(
        '*', count='exact').eq('chatbot_id', chatbot_id)
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
        all_ids = set()
        offset = 0
        while True:
            response = supabase.table('testing_zaps2').select(
                'chatbot_id').range(offset, offset + 999).execute()
            if not response.data:
                break

            batch_ids = set(item['chatbot_id'] for item in response.data
                            if item.get('chatbot_id'))
            all_ids.update(batch_ids)

            if len(response.data) < 1000:
                break

            offset += 1000

        return list(all_ids)
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
            if (messages[i].get("role") == "assistant"
                    and unanswered_message in messages[i].get("content", "")):
                if i > 0 and messages[i - 1].get("role") == "user":
                    unanswered_queries.append(messages[i - 1].get(
                        "content", ""))

    return unanswered_queries


def clean_existing_plots(chatbot_id, period):
    """Clean existing plots for the given chatbot and period"""
    PLOT_BUCKET_NAME = "plots"
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = f"{chatbot_id}/{today}/{period}"
    
    try:
        # List all files in the directory
        response = supabase.storage.from_(PLOT_BUCKET_NAME).list(path=dir_path)
        if response:
            # Delete each file
            files_to_delete = [f"{dir_path}/{file['name']}" for file in response]
            if files_to_delete:
                supabase.storage.from_(PLOT_BUCKET_NAME).remove(files_to_delete)
                print(f"Cleaned {len(files_to_delete)} existing plots for {chatbot_id}/{period}")
    except Exception as e:
        print(f"Note: Could not clean existing plots (this is normal for first run): {e}")


def save_plot_to_supabase(plt, plot_name, chatbot_id, period):
    PLOT_BUCKET_NAME = "plots"
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = f"{chatbot_id}/{today}/{period}"

    try:
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        file_content = buf.read()
        buf.close()

        # Upload to Supabase with full path
        filename = f"{dir_path}/{plot_name}.png"
        response = supabase.storage.from_(PLOT_BUCKET_NAME).upload(
            filename, file_content, {
                "contentType": "image/png",
                "upsert": "true"
            })
        if not response:
            print(f"Error uploading {plot_name} to Supabase")

    except Exception as e:
        print(f"Error saving plot to Supabase: {e}")





def generate_message_distribution(user_queries, assistant_responses,
                                  chatbot_id, period):
    try:
        labels = ["USER QUERIES", "ASSISTANT RESPONSES"]
        sizes = [len(user_queries), len(assistant_responses)]
        if not sizes or sum(sizes) == 0:
            sizes = [1, 1]
        plt.figure(figsize=(10, 10), dpi=150)
        plt.pie(sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#ff9999", "#66b3ff"])
        save_plot_to_supabase(plt, "Message distribution plot", chatbot_id,
                              period)
    except Exception as e:
        print(f"Error generating message distribution: {e}")
    finally:
        plt.close()


def generate_sentiment_analysis(user_queries, chatbot_id, period):
    try:
        if not user_queries:
            plt.figure(figsize=(12, 8), dpi=150)
            plt.text(0.5, 0.5, 'No queries available for sentiment analysis',
                     horizontalalignment='center', fontsize=16)
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

        # Create enhanced donut chart if we have data
        if sum(sentiments.values()) > 0:
            plt.figure(figsize=(14, 10), dpi=150)
            
            # Reorder for better visual flow: Positive, Neutral, Negative
            ordered_labels = ["Positive", "Neutral", "Negative"]
            ordered_values = [sentiments[label] for label in ordered_labels]
            colors = ["#27AE60", "#F39C12", "#E74C3C"]  # Enhanced colors
            
            # Create donut chart with enhanced styling
            wedges, texts, autotexts = plt.pie(
                ordered_values, 
                labels=ordered_labels, 
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(ordered_values))})',
                startangle=90,
                pctdistance=0.75,
                labeldistance=1.15,
                textprops={'fontsize': 13, 'fontweight': 'bold'},
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3)
            )
            
            # Style the percentage labels
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
                autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.6))
            
            # Style the category labels
            for text in texts:
                text.set_fontsize(14)
                text.set_fontweight('bold')
            
            # Add center content
            total_queries = sum(ordered_values)
            plt.text(0, 0.1, 'TOTAL', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='#2C3E50')
            plt.text(0, -0.1, f'{total_queries:,}', ha='center', va='center', 
                    fontsize=24, fontweight='bold', color='#2C3E50')
            plt.text(0, -0.25, 'QUERIES', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='#7F8C8D')
            
            # Create detailed legend with counts and percentages
            legend_elements = []
            for i, (label, value) in enumerate(zip(ordered_labels, ordered_values)):
                percentage = (value / total_queries) * 100
                legend_elements.append(f'{label}: {value:,} ({percentage:.1f}%)')
            
            plt.legend(wedges, legend_elements, 
                      title="Sentiment Breakdown", 
                      loc="center left", 
                      bbox_to_anchor=(1.1, 0, 0.5, 1),
                      fontsize=12, 
                      title_fontsize=14,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
            
            # Enhanced title with subtitle
            plt.suptitle('User Query Sentiment Analysis', 
                        fontsize=20, fontweight='bold', y=0.95, color='#2C3E50')
            plt.title('Distribution of Emotional Tone in User Interactions', 
                     fontsize=14, style='italic', color='#7F8C8D', pad=20)
            
            # Add insights box
            dominant_sentiment = ordered_labels[ordered_values.index(max(ordered_values))]
            dominant_percentage = (max(ordered_values) / total_queries) * 100
            
            insight_text = f"📊 Insights:\n• Dominant sentiment: {dominant_sentiment} ({dominant_percentage:.1f}%)\n• Sample size: {total_queries:,} queries"
            plt.text(1.15, 0.85, insight_text,
                    transform=plt.gca().transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECF0F1", alpha=0.9, edgecolor='#BDC3C7'))
            
            plt.axis('equal')
            plt.tight_layout()
            
        else:
            plt.figure(figsize=(14, 10), dpi=150)
            plt.text(0.5, 0.5, 'No sentiment data available',
                     horizontalalignment='center', fontsize=16)
            plt.axis('off')

        save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
    except Exception as e:
        print(f"Error generating sentiment analysis: {e}")
    finally:
        plt.close()


def generate_chat_volume_plot(conversations, chatbot_id, period):
    try:
        if period == 0:
            # For all-time data, show monthly aggregation
            monthly_counts = {}
            for convo in conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    month_key = convo_date.strftime("%Y-%m")
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                except Exception as e:
                    continue

            if not monthly_counts:
                plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, 'No conversation data available', 
                        horizontalalignment='center', fontsize=20)
                plt.axis('off')
            else:
                dates = sorted(monthly_counts.keys())
                counts = [monthly_counts[date] for date in dates]

                plt.figure(figsize=(14, 8), dpi=150)
                plt.plot(dates, counts, marker='o', linewidth=2, markersize=6, color='#2E86C1')
                plt.xlabel('Month', fontsize=14)
                plt.ylabel('Number of Conversations', fontsize=14)
                plt.title('Chat Volume Over Time (Monthly)', fontsize=18, pad=20)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
        else:
            # For specific periods, show daily data
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_counts = {}

            # Initialize all days in the period with 0 count
            for i in range(period):
                date_key = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_counts[date_key] = 0

            # Count conversations for each day
            for convo in conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    date_key = convo["date_of_convo"]
                    if date_key in daily_counts:
                        daily_counts[date_key] += 1
                except Exception as e:
                    continue

            # Sort dates and get corresponding counts
            dates = sorted(daily_counts.keys())
            counts = [daily_counts[date] for date in dates]

            # Format dates for display
            formatted_dates = [datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d") for date in dates]

            plt.figure(figsize=(14, 8), dpi=150)
            if sum(counts) == 0:
                plt.text(0.5, 0.5, f'No conversations in the last {period} days', 
                        horizontalalignment='center', fontsize=20)
                plt.axis('off')
            else:
                plt.plot(formatted_dates, counts, marker='o', linewidth=2, markersize=6, color='#2E86C1')
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Number of Conversations', fontsize=14)
                plt.title(f'Chat Volume Over Time (Last {period} Days)', fontsize=18, pad=20)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Add trend information
                if len(counts) > 1:
                    # Calculate percentage change from first to last day
                    first_count = counts[0] if counts[0] > 0 else 1
                    last_count = counts[-1]
                    pct_change = ((last_count - first_count) / first_count) * 100

                    trend_text = f"Trend: {pct_change:+.1f}% change"
                    plt.text(0.02, 0.98, trend_text, transform=plt.gca().transAxes, 
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        save_plot_to_supabase(plt, "Chat volume plot", chatbot_id, period)
    except Exception as e:
        print(f"Error generating chat volume plot: {e}")
    finally:
        plt.close()


def generate_peak_hours_activity_plot(conversations, chatbot_id, period):
    try:
        # Initialize 24-hour activity counter
        hourly_activity = {hour: 0 for hour in range(24)}

        # Count conversations by hour
        for convo in conversations:
            if "messages" not in convo:
                continue

            messages = convo["messages"]
            for message in messages:
                if message.get("role") == "user":
                    # Extract hour from created_at or use a default pattern
                    try:
                        # Try to get timestamp from message or conversation
                        if "created_at" in convo:
                            timestamp = datetime.fromisoformat(convo["created_at"].replace('Z', '+00:00'))
                            hour = timestamp.hour
                            hourly_activity[hour] += 1
                        break  # Only count first user message per conversation
                    except Exception as e:
                        continue

        # Create the plot
        hours = list(range(24))
        activity_counts = [hourly_activity[hour] for hour in hours]

        plt.figure(figsize=(15, 8), dpi=150)

        if sum(activity_counts) == 0:
            plt.text(0.5, 0.5, 'No activity data available for peak hours analysis', 
                    horizontalalignment='center', fontsize=16)
            plt.axis('off')
        else:
            # Create bar chart
            bars = plt.bar(hours, activity_counts, color='#3498db', alpha=0.7, edgecolor='#2980b9')

            # Highlight peak hours
            peak_hour = hours[activity_counts.index(max(activity_counts))]
            bars[peak_hour].set_color('#e74c3c')

            # Format the plot
            plt.xlabel('Hour of Day (24-hour format)', fontsize=14)
            plt.ylabel('Number of Conversations', fontsize=14)
            plt.title(f'Peak Hours Activity Analysis ({period} days)' if period > 0 else 'Peak Hours Activity Analysis (All Time)', 
                     fontsize=18, pad=20)

            # Set x-axis ticks and labels with better spacing
            plt.xticks(range(0, 24, 2), [f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            # Adjust y-axis to give more space for annotations
            plt.ylim(0, max(activity_counts) * 1.3)

            # Add peak hour annotation
            peak_count = max(activity_counts)
            plt.annotate(f'Peak Hour: {peak_hour:02d}:00\n({peak_count} conversations)', 
                        xy=(peak_hour, peak_count), 
                        xytext=(peak_hour + 2, peak_count + max(activity_counts) * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=12, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

            # Add insights
            total_conversations = sum(activity_counts)
            business_hours_activity = sum(activity_counts[9:17])  # 9 AM to 5 PM
            business_hours_percentage = (business_hours_activity / total_conversations * 100) if total_conversations > 0 else 0

            insights_text = f"Business Hours (9AM-5PM): {business_hours_percentage:.1f}% of activity"
            plt.text(0.02, 0.98, insights_text, transform=plt.gca().transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

            plt.tight_layout()

        save_plot_to_supabase(plt, "Peak hours activity plot", chatbot_id, period)

    except Exception as e:
        print(f"Error generating peak hours activity plot: {e}")
    finally:
        plt.close()





def get_conversation_insights(chatbot_id, period):
    try:
        processed_ids = set()
        all_conversations = fetch_all_conversations(chatbot_id)

        filtered_conversations = []
        if period > 0:
            current_date = datetime.now().replace(hour=0,
                                                  minute=0,
                                                  second=0,
                                                  microsecond=0)
            cutoff_date = current_date - timedelta(days=period)

            for convo in all_conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"],
                                                   "%Y-%m-%d")
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
        unanswered_query_counts = dict(
            Counter(unanswered_queries).most_common())

        unanswered_queries_json = {
            "queries": unanswered_query_counts,
            "total_count": len(unanswered_queries)
        }

        top_user_queries_dict = dict(Counter(user_queries).most_common(10))
        top_user_queries_json = {"queries": top_user_queries_dict}

        # Clean existing plots first to ensure fresh generation
        clean_existing_plots(chatbot_id, period)
        
        # Generate all visualizations (Top 10 User Queries plot and Monthly Heatmap removed)
        generate_message_distribution(user_queries, assistant_responses,
                                      chatbot_id, period)
        generate_sentiment_analysis(user_queries, chatbot_id, period)
        generate_chat_volume_plot(conversations, chatbot_id, period)
        generate_peak_hours_activity_plot(conversations, chatbot_id, period)

        # Prepare insights data
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
            "created_at": datetime.now().isoformat()
        }

        # Check if a row with the same chatbot_id, period_range, and date_of_convo already exists
        existing_record = supabase.table('insights') \
            .select('id') \
            .eq('chatbot_id', chatbot_id) \
            .eq('period_range', period) \
            .eq('date_of_convo', current_date) \
            .execute()

        if existing_record.data:
            # Update existing record
            record_id = existing_record.data[0]['id']
            supabase.table('insights') \
                .update(insights_data) \
                .eq('id', record_id) \
                .execute()
            print(f"Updated existing insights record for chatbot {chatbot_id}, period {period}, date {current_date}")
        else:
            # Insert new record
            supabase.table('insights').insert(insights_data).execute()
            print(f"Inserted new insights record for chatbot {chatbot_id}, period {period}, date {current_date}")

        return insights_data
    except Exception as e:
        print(
            f"Error generating conversation insights for chatbot {chatbot_id}: {e}"
        )


def update_tokens():
    try:
        print("\nStarting token update process")
        chatbot_ids = get_distinct_chatbot_ids()
        today = datetime.now().date().isoformat()

        for chatbot_id in chatbot_ids:
            try:
                # Get today's conversations for this chatbot
                response = supabase.table('testing_zaps2') \
                    .select('input_tokens, output_tokens') \
                    .eq('chatbot_id', chatbot_id) \
                    .eq('date_of_convo', today) \
                    .execute()

                if not response.data:
                    continue

                # Calculate total tokens
                # Convert None to 0 when summing tokens
                total_input_tokens = sum(
                    int(row.get('input_tokens') or 0) for row in response.data)
                total_output_tokens = sum(
                    int(row.get('output_tokens') or 0)
                    for row in response.data)

                # Check if chatbot exists in chat_tokens
                existing_tokens = supabase.table('chat_tokens') \
                    .select('input_tokens, output_tokens') \
                    .eq('chatbot_id', chatbot_id) \
                    .execute()

                if existing_tokens.data:
                    # Update existing record with safe conversion of None to 0
                    current_input = int(
                        existing_tokens.data[0].get('input_tokens') or 0)
                    current_output = int(
                        existing_tokens.data[0].get('output_tokens') or 0)
                    # No need for additional conversion since we already have integers

                    supabase.table('chat_tokens') \
                        .update({
                            'input_tokens': current_input + total_input_tokens,
                            'output_tokens': current_output + total_output_tokens
                        }) \
                        .eq('chatbot_id', chatbot_id) \
                        .execute()
                else:
                    # Create new record
                    supabase.table('chat_tokens') \
                        .insert({
                            'chatbot_id': chatbot_id,
                            'input_tokens': total_input_tokens,
                            'output_tokens': total_output_tokens
                        }) \
                        .execute()

                print(f"Updated tokens for chatbot {chatbot_id}")

            except Exception as e:
                print(f"Error updating tokens for chatbot {chatbot_id}: {e}")
                continue

        print("Token update process completed")
    except Exception as e:
        print(f"Error in update_tokens: {e}")


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
                    print(
                        f"Successfully generated insights for {chatbot_id} ({period} days)"
                    )
                    print(
                        f"Total conversations: {insights['total_conversations']}"
                    )
                    print(f"Total queries: {insights['total_user_queries']}")

        except Exception as e:
            print(f"Error processing chatbot ID {chatbot_id}: {e}")
            continue

    print(f"\nCompleted data processing at {datetime.now()}")


def update_job_status(job_statuses):
    try:
        # Get current timestamp
        current_time = datetime.now().isoformat()

        # Create status entry
        status_data = {
            "created_at": current_time,
            "job_status": job_statuses
        }

        # Update insights_schedule table
        supabase.table('insights_schedule').insert(status_data).execute()
        print("Job statuses updated successfully")
    except Exception as e:
        print(f"Error updating job status: {e}")

if __name__ == "__main__":
    job_statuses = []
    try:
        # Run token update
        try:
            update_tokens()
            job_statuses.append({
                "job_name": "update_tokens",
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "error": None
            })
        except Exception as e:
            job_statuses.append({
                "job_name": "update_tokens", 
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        # Run data processing
        try:
            process_chatbot_data()
            job_statuses.append({
                "job_name": "process_chatbot_data",
                "status": "success", 
                "timestamp": datetime.now().isoformat(),
                "error": None
            })
        except Exception as e:
            job_statuses.append({
                "job_name": "process_chatbot_data",
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })

        # Update job statuses in insights_schedule table
        update_job_status(job_statuses)

    except Exception as e:
        print(f"Error in main execution: {e}")
        # Record overall execution failure
        job_statuses.append({
            "job_name": "main_execution",
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })
        update_job_status(job_statuses)