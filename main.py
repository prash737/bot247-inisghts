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
                print(f"Cleaned {len(files_to_delete)} existing plots for {chatbot_id}/{period} (expecting 6 plots total)")
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





def generate_message_distribution(user_queries, assistant_responses, chatbot_id, period):
    try:
        if not user_queries and not assistant_responses:
            plt.figure(figsize=(16, 12), dpi=150)
            plt.text(0.5, 0.5, 'No conversation data available for analysis', 
                    horizontalalignment='center', fontsize=18)
            plt.axis('off')
            save_plot_to_supabase(plt, "Message distribution plot", chatbot_id, period)
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), dpi=150)
        fig.suptitle('Advanced Conversation Flow & Message Analysis', fontsize=24, fontweight='bold', y=0.98)

        # 1. Enhanced Message Distribution (top-left)
        total_queries = len(user_queries)
        total_responses = len(assistant_responses)
        
        # Calculate message complexity
        avg_query_length = np.mean([len(q.split()) for q in user_queries]) if user_queries else 0
        avg_response_length = np.mean([len(r.split()) for r in assistant_responses]) if assistant_responses else 0
        
        labels = ['User Queries', 'Assistant Responses']
        sizes = [total_queries, total_responses]
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                          wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
        
        # Add center information
        ax1.text(0, 0.1, f'{total_queries + total_responses:,}', ha='center', va='center', 
                fontsize=18, fontweight='bold', color='#2C3E50')
        ax1.text(0, -0.1, 'Total Messages', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='#7F8C8D')
        
        ax1.set_title('Message Volume Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Message Length Analysis (top-right)
        if user_queries and assistant_responses:
            query_lengths = [len(q.split()) for q in user_queries]
            response_lengths = [len(r.split()) for r in assistant_responses]
            
            ax2.hist(query_lengths, bins=20, alpha=0.7, label='User Queries', color='#FF6B6B', density=True)
            ax2.hist(response_lengths, bins=20, alpha=0.7, label='Assistant Responses', color='#4ECDC4', density=True)
            
            ax2.axvline(np.mean(query_lengths), color='#FF6B6B', linestyle='--', linewidth=2, 
                       label=f'Avg Query: {avg_query_length:.1f} words')
            ax2.axvline(np.mean(response_lengths), color='#4ECDC4', linestyle='--', linewidth=2, 
                       label=f'Avg Response: {avg_response_length:.1f} words')
            
            ax2.set_xlabel('Message Length (words)', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Message Length Distribution Analysis', fontsize=16, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for length analysis', ha='center', va='center', fontsize=14)
            ax2.set_title('Message Length Distribution', fontsize=16, fontweight='bold')
        
        # 3. Conversation Complexity Index (bottom-left)
        complexity_categories = {
            'Simple (1-5 words)': 0,
            'Moderate (6-15 words)': 0,
            'Complex (16-30 words)': 0,
            'Very Complex (31+ words)': 0
        }
        
        all_messages = user_queries + assistant_responses
        for msg in all_messages:
            word_count = len(msg.split())
            if word_count <= 5:
                complexity_categories['Simple (1-5 words)'] += 1
            elif word_count <= 15:
                complexity_categories['Moderate (6-15 words)'] += 1
            elif word_count <= 30:
                complexity_categories['Complex (16-30 words)'] += 1
            else:
                complexity_categories['Very Complex (31+ words)'] += 1
        
        categories = list(complexity_categories.keys())
        values = list(complexity_categories.values())
        colors_complexity = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
        
        bars = ax3.bar(categories, values, color=colors_complexity, alpha=0.8)
        ax3.set_xlabel('Complexity Level', fontsize=12)
        ax3.set_ylabel('Number of Messages', fontsize=12)
        ax3.set_title('Message Complexity Distribution', fontsize=16, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. Conversation Flow Metrics (bottom-right)
        metrics_data = {
            'Query-Response Ratio': f'{total_responses/total_queries:.2f}' if total_queries > 0 else 'N/A',
            'Avg Query Length': f'{avg_query_length:.1f} words',
            'Avg Response Length': f'{avg_response_length:.1f} words',
            'Verbosity Index': f'{avg_response_length/avg_query_length:.2f}x' if avg_query_length > 0 else 'N/A',
            'Total Word Count': f'{sum(len(msg.split()) for msg in all_messages):,} words',
            'Conversation Density': f'{(total_queries + total_responses)/max(1, len(set(all_messages))):.1f} msgs/topic'
        }
        
        # Create a clean metrics display
        ax4.axis('off')
        ax4.set_title('Conversation Flow Metrics', fontsize=16, fontweight='bold', pad=20)
        
        y_positions = np.linspace(0.8, 0.1, len(metrics_data))
        for i, (metric, value) in enumerate(metrics_data.items()):
            # Metric name
            ax4.text(0.05, y_positions[i], metric + ':', fontsize=13, fontweight='bold', 
                    transform=ax4.transAxes, verticalalignment='center')
            # Metric value
            ax4.text(0.95, y_positions[i], str(value), fontsize=13, 
                    transform=ax4.transAxes, verticalalignment='center', 
                    horizontalalignment='right', color='#2C3E50',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", alpha=0.7))
        
        # Add insights box
        if total_queries > 0:
            response_rate = (total_responses / total_queries) * 100
            insight_color = '#27AE60' if response_rate > 90 else '#F39C12' if response_rate > 70 else '#E74C3C'
            
            insight_text = f"📊 Key Insights:\n"
            insight_text += f"• Response Rate: {response_rate:.1f}%\n"
            insight_text += f"• Communication Style: {'Detailed' if avg_response_length > avg_query_length * 2 else 'Concise'}\n"
            insight_text += f"• Interaction Quality: {'High' if response_rate > 85 else 'Moderate' if response_rate > 70 else 'Needs Improvement'}"
            
            ax4.text(0.5, 0.02, insight_text, transform=ax4.transAxes, fontsize=11, 
                    horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=insight_color, alpha=0.1, 
                             edgecolor=insight_color, linewidth=2))
        
        plt.tight_layout()
        save_plot_to_supabase(plt, "Message distribution plot", chatbot_id, period)
        
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
            plt.figure(figsize=(16, 12), dpi=150)

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
                labeldistance=1.2,
                textprops={'fontsize': 14, 'fontweight': 'bold'},
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3)
            )

            # Style the percentage labels
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(13)
                autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            # Style the category labels
            for text in texts:
                text.set_fontsize(16)
                text.set_fontweight('bold')

            # Add center content
            total_queries = sum(ordered_values)
            plt.text(0, 0.15, 'TOTAL', ha='center', va='center', 
                    fontsize=18, fontweight='bold', color='#2C3E50')
            plt.text(0, 0, f'{total_queries:,}', ha='center', va='center', 
                    fontsize=28, fontweight='bold', color='#2C3E50')
            plt.text(0, -0.15, 'QUERIES', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='#7F8C8D')

            # Enhanced title - positioned higher to avoid overlap
            plt.title('User Query Sentiment Analysis', 
                     fontsize=22, fontweight='bold', y=1.08, color='#2C3E50', pad=20)

            # Create detailed legend positioned better
            legend_elements = []
            for i, (label, value) in enumerate(zip(ordered_labels, ordered_values)):
                percentage = (value / total_queries) * 100
                legend_elements.append(f'{label}: {value:,} ({percentage:.1f}%)')

            plt.legend(wedges, legend_elements, 
                      title="Sentiment Breakdown", 
                      loc="center left", 
                      bbox_to_anchor=(1.05, 0.5),
                      fontsize=13, 
                      title_fontsize=16,
                      frameon=True,
                      fancybox=True,
                      shadow=True)

            # Add insights box in a better position
            dominant_sentiment = ordered_labels[ordered_values.index(max(ordered_values))]
            dominant_percentage = (max(ordered_values) / total_queries) * 100

            insight_text = f"Key Insights:\n• Dominant sentiment: {dominant_sentiment} ({dominant_percentage:.1f}%)\n• Total analyzed: {total_queries:,} queries\n• Distribution shows emotional tone trends"
            plt.text(1.05, 0.15, insight_text,
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECF0F1", alpha=0.9, edgecolor='#BDC3C7', linewidth=1))

            plt.axis('equal')
            plt.subplots_adjust(left=0.1, right=0.75, top=0.85, bottom=0.1)

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

            # Highlight peak hour
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


def generate_response_time_analysis(conversations, chatbot_id, period):
    try:
        response_times = []

        for convo in conversations:
            if "messages" not in convo:
                continue

            messages = convo["messages"]
            for i in range(len(messages) - 1):
                if (messages[i].get("role") == "user" and 
                    messages[i + 1].get("role") == "assistant"):
                    # Simulate response time analysis (in real scenario, you'd have timestamps)
                    # For demo, we'll use message length as a proxy for complexity/response time
                    user_msg_length = len(messages[i].get("content", ""))
                    assistant_msg_length = len(messages[i + 1].get("content", ""))

                    # Simple heuristic: longer responses take more time
                    estimated_response_time = min(max(assistant_msg_length / 20, 0.5), 30)  # 0.5-30 seconds
                    response_times.append(estimated_response_time)

        plt.figure(figsize=(14, 8), dpi=150)

        if not response_times:
            plt.text(0.5, 0.5, 'No response time data available', 
                    horizontalalignment='center', fontsize=16)
            plt.axis('off')
        else:
            # Create histogram
            plt.hist(response_times, bins=20, color='#9b59b6', alpha=0.7, edgecolor='#8e44ad')

            # Add statistics
            avg_response_time = np.mean(response_times)
            median_response_time = np.median(response_times)

            plt.axvline(avg_response_time, color='red', linestyle='--', linewidth=2, 
                       label=f'Average: {avg_response_time:.1f}s')
            plt.axvline(median_response_time, color='orange', linestyle='--', linewidth=2, 
                       label=f'Median: {median_response_time:.1f}s')

            plt.xlabel('Response Time (seconds)', fontsize=14)
            plt.ylabel('Number of Responses', fontsize=14)
            plt.title(f'Response Time Analysis ({period} days)' if period > 0 else 'Response Time Analysis (All Time)', 
                     fontsize=18, pad=20)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add performance insights
            fast_responses = sum(1 for rt in response_times if rt <= 2)
            fast_percentage = (fast_responses / len(response_times)) * 100

            insights_text = f"Performance Insights:\n• Fast responses (≤2s): {fast_percentage:.1f}%\n• Total responses analyzed: {len(response_times):,}\n• Average response time: {avg_response_time:.1f}s"
            plt.text(0.98, 0.98, insights_text, transform=plt.gca().transAxes, 
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", alpha=0.9, edgecolor='#1ABC9C'))

            plt.tight_layout()

        save_plot_to_supabase(plt, "Response time analysis plot", chatbot_id, period)

    except Exception as e:
        print(f"Error generating response time analysis: {e}")
    finally:
        plt.close()


def generate_user_engagement_funnel(conversations, chatbot_id, period):
    try:
        if not conversations:
            plt.figure(figsize=(16, 12), dpi=150)
            plt.text(0.5, 0.5, 'No engagement data available', 
                    horizontalalignment='center', fontsize=16)
            plt.axis('off')
            save_plot_to_supabase(plt, "User engagement funnel plot", chatbot_id, period)
            return

        # Advanced engagement metrics
        engagement_stages = {
            "Initial Contact": 0,
            "Engaged (2-4 msgs)": 0,
            "Active (5-9 msgs)": 0,
            "Highly Engaged (10-19 msgs)": 0,
            "Power Users (20+ msgs)": 0
        }

        conversation_data = []
        session_durations = []
        message_intervals = []

        for convo in conversations:
            if "messages" not in convo:
                continue

            messages = convo["messages"]
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            user_message_count = len(user_messages)
            
            if user_message_count == 0:
                continue

            conversation_data.append({
                'user_messages': user_message_count,
                'total_messages': len(messages),
                'conversation_id': convo.get('id', 'unknown')
            })

            # Categorize engagement levels
            if user_message_count >= 1:
                engagement_stages["Initial Contact"] += 1
            if 2 <= user_message_count <= 4:
                engagement_stages["Engaged (2-4 msgs)"] += 1
            elif 5 <= user_message_count <= 9:
                engagement_stages["Active (5-9 msgs)"] += 1
            elif 10 <= user_message_count <= 19:
                engagement_stages["Highly Engaged (10-19 msgs)"] += 1
            elif user_message_count >= 20:
                engagement_stages["Power Users (20+ msgs)"] += 1

        if not conversation_data:
            plt.figure(figsize=(16, 12), dpi=150)
            plt.text(0.5, 0.5, 'No valid conversation data available', 
                    horizontalalignment='center', fontsize=16)
            plt.axis('off')
            save_plot_to_supabase(plt, "User engagement funnel plot", chatbot_id, period)
            return

        # Create comprehensive engagement dashboard
        fig = plt.figure(figsize=(20, 16), dpi=150)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main Engagement Funnel (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        stages = list(engagement_stages.keys())
        values = list(engagement_stages.values())
        colors = ['#1ABC9C', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C']
        
        # Create funnel effect with trapezoids
        y_positions = np.arange(len(stages))[::-1]  # Reverse order for funnel effect
        
        for i, (stage, value, color) in enumerate(zip(stages, values, colors)):
            if value > 0:
                # Calculate width for funnel effect
                max_width = max(values)
                width = (value / max_width) * 0.8 + 0.2  # Minimum 20% width
                
                # Create trapezoid-like bars
                ax1.barh(y_positions[i], value, height=0.6, color=color, alpha=0.8, 
                        edgecolor='white', linewidth=2)
                
                # Add labels
                ax1.text(value + max(values) * 0.02, y_positions[i], 
                        f'{value:,} users ({value/values[0]*100:.1f}%)', 
                        va='center', fontsize=12, fontweight='bold')

        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(stages, fontsize=12)
        ax1.set_xlabel('Number of Users', fontsize=14, fontweight='bold')
        ax1.set_title('Advanced User Engagement Funnel', fontsize=18, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. Engagement Distribution Pie Chart (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Filter out zero values for pie chart
        non_zero_stages = [(stage, value) for stage, value in engagement_stages.items() if value > 0]
        if non_zero_stages:
            pie_labels, pie_values = zip(*non_zero_stages)
            pie_colors = [colors[stages.index(label)] for label in pie_labels]
            
            wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, colors=pie_colors,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 10},
                                              wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))
            
            # Add center text
            total_users = sum(pie_values)
            ax2.text(0, 0, f'{total_users:,}\nUsers', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='#2C3E50')
        
        ax2.set_title('Engagement Level Distribution', fontsize=14, fontweight='bold')

        # 3. Message Count Distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        user_message_counts = [data['user_messages'] for data in conversation_data]
        
        ax3.hist(user_message_counts, bins=20, color='#3498DB', alpha=0.7, edgecolor='white')
        ax3.axvline(np.mean(user_message_counts), color='#E74C3C', linestyle='--', linewidth=2,
                   label=f'Average: {np.mean(user_message_counts):.1f}')
        ax3.axvline(np.median(user_message_counts), color='#F39C12', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(user_message_counts):.1f}')
        
        ax3.set_xlabel('Messages per User', fontsize=12)
        ax3.set_ylabel('Number of Users', fontsize=12)
        ax3.set_title('User Message Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Engagement Quality Matrix (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create engagement quality metrics
        quality_metrics = {
            'Bounce Rate': f"{(values[0] - sum(values[1:]))/values[0]*100:.1f}%" if values[0] > 0 else "N/A",
            'Conversion to Active': f"{sum(values[2:])/values[0]*100:.1f}%" if values[0] > 0 else "N/A",
            'Retention Score': f"{sum(values[3:])/values[0]*100:.1f}%" if values[0] > 0 else "N/A",
            'Power User Rate': f"{values[4]/values[0]*100:.1f}%" if values[0] > 0 else "N/A"
        }
        
        ax4.axis('off')
        ax4.set_title('Engagement Quality Metrics', fontsize=14, fontweight='bold', pad=20)
        
        y_pos = 0.8
        for metric, value in quality_metrics.items():
            ax4.text(0.1, y_pos, f"• {metric}:", fontsize=12, fontweight='bold', 
                    transform=ax4.transAxes)
            ax4.text(0.9, y_pos, value, fontsize=12, 
                    transform=ax4.transAxes, ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", alpha=0.8))
            y_pos -= 0.15

        # 5. Conversation Length Trends (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Group conversations by length ranges
        length_ranges = {'1': 0, '2-4': 0, '5-9': 0, '10-19': 0, '20+': 0}
        for count in user_message_counts:
            if count == 1:
                length_ranges['1'] += 1
            elif 2 <= count <= 4:
                length_ranges['2-4'] += 1
            elif 5 <= count <= 9:
                length_ranges['5-9'] += 1
            elif 10 <= count <= 19:
                length_ranges['10-19'] += 1
            else:
                length_ranges['20+'] += 1
        
        ranges = list(length_ranges.keys())
        counts = list(length_ranges.values())
        
        bars = ax5.bar(ranges, counts, color=['#95A5A6', '#3498DB', '#2ECC71', '#F39C12', '#E74C3C'], alpha=0.8)
        ax5.set_xlabel('Message Count Range', fontsize=12)
        ax5.set_ylabel('Number of Users', fontsize=12)
        ax5.set_title('Conversation Length Segments', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontweight='bold')

        # 6. Advanced Insights Panel (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate advanced metrics
        total_conversations = len(conversation_data)
        avg_messages_per_user = np.mean(user_message_counts)
        engagement_score = (sum(values[2:]) / values[0] * 100) if values[0] > 0 else 0
        
        # Determine engagement level
        if engagement_score >= 40:
            engagement_level = "Excellent"
            level_color = "#27AE60"
        elif engagement_score >= 25:
            engagement_level = "Good"
            level_color = "#F39C12"
        elif engagement_score >= 15:
            engagement_level = "Fair"
            level_color = "#E67E22"
        else:
            engagement_level = "Needs Improvement"
            level_color = "#E74C3C"

        insights_text = f"""
🎯 ENGAGEMENT INSIGHTS & RECOMMENDATIONS

📊 Overall Performance:
   • Total Conversations: {total_conversations:,}
   • Average Messages/User: {avg_messages_per_user:.1f}
   • Engagement Score: {engagement_score:.1f}% ({engagement_level})
   • User Retention: {(sum(values[1:]) / values[0] * 100):.1f}% continue past first interaction

🔍 Key Findings:
   • {values[4]} power users (20+ messages) - these are your most valuable users
   • {values[0] - sum(values[1:])} users dropped after first interaction
   • {sum(values[2:4])} users show strong engagement potential

💡 Actionable Recommendations:
   • Focus on reducing bounce rate from {(values[0] - sum(values[1:]))/values[0]*100:.1f}%
   • Implement engagement triggers for users with 2-4 messages
   • Analyze power user conversations to identify success patterns
        """

        ax6.text(0.02, 0.98, insights_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=level_color, alpha=0.1, 
                         edgecolor=level_color, linewidth=2))

        fig.suptitle(f'User Engagement Analytics Dashboard ({period} days)' if period > 0 else 'User Engagement Analytics Dashboard (All Time)', 
                     fontsize=24, fontweight='bold', y=0.98)

        save_plot_to_supabase(plt, "User engagement funnel plot", chatbot_id, period)

    except Exception as e:
        print(f"Error generating user engagement funnel: {e}")
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
        generate_response_time_analysis(conversations, chatbot_id, period)
        generate_user_engagement_funnel(conversations, chatbot_id, period)


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