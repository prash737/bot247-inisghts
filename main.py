
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
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
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
    """Message distribution pie chart"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not user_queries and not assistant_responses:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            total_queries = len(user_queries)
            total_responses = len(assistant_responses)
            
            labels = ['User Queries', 'Assistant Responses']
            sizes = [total_queries, total_responses]
            colors = ['#3498DB', '#E74C3C']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90,
                                              wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3),
                                              textprops={'fontsize': 14, 'fontweight': 'bold'})
            
            # Center text
            ax.text(0, 0.1, f'{total_queries + total_responses:,}', ha='center', va='center', 
                    fontsize=20, fontweight='bold', color='#2C3E50')
            ax.text(0, -0.1, 'Total Messages', ha='center', va='center', 
                    fontsize=12, color='#7F8C8D')
        
        ax.set_title('Message Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Message distribution plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating message distribution: {e}")
    finally:
        plt.close()


def generate_message_length_analysis(user_queries, assistant_responses, chatbot_id, period):
    """Message length distribution analysis"""
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
            query_lengths = [len(q.split()) for q in user_queries]
            response_lengths = [len(r.split()) for r in assistant_responses]
            
            # Create bins for better visualization
            max_length = max(max(query_lengths) if query_lengths else 0, 
                           max(response_lengths) if response_lengths else 0)
            bins = np.linspace(0, min(max_length, 100), 25)
            
            ax.hist(query_lengths, bins=bins, alpha=0.7, label='User Queries', 
                    color='#3498DB', density=True, edgecolor='white')
            ax.hist(response_lengths, bins=bins, alpha=0.7, label='Assistant Responses', 
                    color='#E74C3C', density=True, edgecolor='white')
            
            avg_query = np.mean(query_lengths)
            avg_response = np.mean(response_lengths)
            
            ax.axvline(avg_query, color='#2980B9', linestyle='--', linewidth=2,
                       label=f'Avg Query: {avg_query:.1f} words')
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
        save_plot_to_supabase(plt, "Message length analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating message length analysis: {e}")
    finally:
        plt.close()


def generate_message_complexity_analysis(user_queries, assistant_responses, chatbot_id, period):
    """Message complexity distribution"""
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
            all_messages = user_queries + assistant_responses
            complexity_data = {'Simple\n(1-5 words)': 0, 'Moderate\n(6-15 words)': 0, 
                             'Complex\n(16-30 words)': 0, 'Very Complex\n(31+ words)': 0}
            
            for msg in all_messages:
                word_count = len(msg.split())
                if word_count <= 5:
                    complexity_data['Simple\n(1-5 words)'] += 1
                elif word_count <= 15:
                    complexity_data['Moderate\n(6-15 words)'] += 1
                elif word_count <= 30:
                    complexity_data['Complex\n(16-30 words)'] += 1
                else:
                    complexity_data['Very Complex\n(31+ words)'] += 1
            
            categories = list(complexity_data.keys())
            values = list(complexity_data.values())
            colors_complexity = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
            
            bars = ax.bar(categories, values, color=colors_complexity, alpha=0.8, 
                          edgecolor='white', linewidth=2)
            ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                            f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title('Message Complexity Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Message complexity analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating message complexity analysis: {e}")
    finally:
        plt.close()


def generate_key_performance_metrics(user_queries, assistant_responses, chatbot_id, period):
    """Key performance metrics panel"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        if not user_queries and not assistant_responses:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
        else:
            total_queries = len(user_queries)
            total_responses = len(assistant_responses)
            all_messages = user_queries + assistant_responses
            
            # Calculate metrics
            avg_query_length = np.mean([len(q.split()) for q in user_queries]) if user_queries else 0
            avg_response_length = np.mean([len(r.split()) for r in assistant_responses]) if assistant_responses else 0
            response_ratio = total_responses / total_queries if total_queries > 0 else 0
            verbosity_index = avg_response_length / avg_query_length if avg_query_length > 0 else 0
            
            metrics = [
                ('Total Conversations', f'{len(set(user_queries + assistant_responses)):,}'),
                ('Response Rate', f'{response_ratio:.2f}:1'),
                ('Avg Query Length', f'{avg_query_length:.1f} words'),
                ('Avg Response Length', f'{avg_response_length:.1f} words'),
                ('Verbosity Index', f'{verbosity_index:.2f}x'),
                ('Total Word Count', f'{sum(len(msg.split()) for msg in all_messages):,}')
            ]
            
            y_start = 0.85
            for i, (metric, value) in enumerate(metrics):
                y_pos = y_start - (i * 0.12)
                
                # Metric box
                ax.text(0.05, y_pos, metric, fontsize=16, fontweight='bold', 
                        transform=ax.transAxes, va='center')
                ax.text(0.95, y_pos, value, fontsize=16, transform=ax.transAxes, 
                        va='center', ha='right', color='#2C3E50',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECF0F1", 
                                 edgecolor='#BDC3C7', linewidth=1))
            
            # Performance indicator
            if total_queries > 0:
                performance_score = min(100, (response_ratio * 50) + (min(avg_response_length, 20) * 2.5))
                performance_level = 'Excellent' if performance_score >= 80 else 'Good' if performance_score >= 60 else 'Fair'
                performance_color = '#27AE60' if performance_score >= 80 else '#F39C12' if performance_score >= 60 else '#E74C3C'
                
                ax.text(0.5, 0.15, f'Performance: {performance_level} ({performance_score:.0f}/100)', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=18, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=performance_color, alpha=0.2, 
                                 edgecolor=performance_color, linewidth=2))
        
        ax.set_title('Key Performance Metrics', fontsize=24, fontweight='bold', y=0.95)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Key performance metrics plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating key performance metrics: {e}")
    finally:
        plt.close()


def generate_sentiment_analysis(user_queries, chatbot_id, period):
    """Sentiment distribution pie chart"""
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
            
            for query in user_queries:
                analysis = TextBlob(query)
                if analysis.sentiment.polarity > 0.1:
                    sentiments["Positive"] += 1
                elif analysis.sentiment.polarity < -0.1:
                    sentiments["Negative"] += 1
                else:
                    sentiments["Neutral"] += 1
            
            ordered_labels = ["Positive", "Neutral", "Negative"]
            ordered_values = [sentiments[label] for label in ordered_labels]
            colors = ["#27AE60", "#F39C12", "#E74C3C"]
            
            if sum(ordered_values) > 0:
                wedges, texts, autotexts = ax.pie(
                    ordered_values, labels=ordered_labels, colors=colors,
                    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(ordered_values))})',
                    startangle=90, pctdistance=0.75, labeldistance=1.15,
                    textprops={'fontsize': 14, 'fontweight': 'bold'},
                    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3)
                )
                
                # Style percentage labels
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                
                # Center content
                total_queries = sum(ordered_values)
                ax.text(0, 0.15, 'SENTIMENT', ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='#2C3E50')
                ax.text(0, 0, f'{total_queries:,}', ha='center', va='center', 
                        fontsize=24, fontweight='bold', color='#2C3E50')
                ax.text(0, -0.15, 'QUERIES', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='#7F8C8D')
        
        ax.set_title('Sentiment Analysis', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Sentiment analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating sentiment analysis: {e}")
    finally:
        plt.close()


def generate_sentiment_score_distribution(user_queries, chatbot_id, period):
    """Sentiment score distribution histogram"""
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
                analysis = TextBlob(query)
                sentiment_scores.append(analysis.sentiment.polarity)
            
            ax.hist(sentiment_scores, bins=30, color='#3498DB', alpha=0.7, edgecolor='white', density=True)
            ax.axvline(0, color='#34495E', linestyle='-', linewidth=2, alpha=0.7, label='Neutral Line')
            ax.axvline(np.mean(sentiment_scores), color='#E74C3C', linestyle='--', linewidth=2, 
                       label=f'Average: {np.mean(sentiment_scores):.3f}')
            
            ax.set_xlabel('Sentiment Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Density', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add annotations for sentiment ranges
            ax.text(-0.8, ax.get_ylim()[1]*0.8, 'Very\nNegative', ha='center', va='center', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#E74C3C", alpha=0.3))
            ax.text(0, ax.get_ylim()[1]*0.9, 'Neutral', ha='center', va='center', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F39C12", alpha=0.3))
            ax.text(0.8, ax.get_ylim()[1]*0.8, 'Very\nPositive', ha='center', va='center', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#27AE60", alpha=0.3))
        
        ax.set_title('Sentiment Score Distribution', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Sentiment score distribution plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating sentiment score distribution: {e}")
    finally:
        plt.close()


def generate_chat_volume_plot(conversations, chatbot_id, period):
    """Chat volume trend analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if period == 0:
            # Monthly aggregation for all-time data
            monthly_counts = {}
            for convo in conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    convo_date = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    month_key = convo_date.strftime("%Y-%m")
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                except Exception:
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

                # Main trend line
                ax.plot(formatted_dates, counts, marker='o', linewidth=3, markersize=8, 
                        color='#3498DB', markerfacecolor='#E74C3C', markeredgecolor='white', markeredgewidth=2)
                ax.fill_between(formatted_dates, counts, alpha=0.3, color='#3498DB')
                
                ax.set_ylabel('Number of Conversations', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add peak annotation
                max_idx = counts.index(max(counts))
                ax.annotate(f'Peak: {max(counts):,}', xy=(max_idx, max(counts)), 
                           xytext=(max_idx, max(counts) + max(counts)*0.1),
                           arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
                           fontsize=12, ha='center', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="#E74C3C", alpha=0.3))
                
        else:
            # Daily data for specific periods
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_counts = {}

            for i in range(period):
                date_key = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
                daily_counts[date_key] = 0

            for convo in conversations:
                try:
                    if "date_of_convo" not in convo:
                        continue
                    date_key = convo["date_of_convo"]
                    if date_key in daily_counts:
                        daily_counts[date_key] += 1
                except Exception:
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
                # Line plot with trend
                ax.plot(formatted_dates, counts, marker='o', linewidth=3, markersize=6, 
                        color='#3498DB', markerfacecolor='#E74C3C', markeredgecolor='white', markeredgewidth=2)
                ax.fill_between(formatted_dates, counts, alpha=0.3, color='#3498DB')
                
                ax.set_ylabel('Number of Conversations', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=11)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add trend line
                if len(counts) > 2:
                    z = np.polyfit(range(len(counts)), counts, 1)
                    p = np.poly1d(z)
                    ax.plot(formatted_dates, p(range(len(counts))), "--", color='#E74C3C', 
                            linewidth=2, alpha=0.8, label=f'Trend: {"↗" if z[0] > 0 else "↘"}')
                    ax.legend(fontsize=12)
        
        title = f'Chat Volume Trend (Last {period} Days)' if period > 0 else 'Chat Volume Trend (All Time)'
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Chat volume plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating chat volume plot: {e}")
    finally:
        plt.close()


def generate_peak_hours_activity_plot(conversations, chatbot_id, period):
    """Hourly activity pattern analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        # Initialize 24-hour activity counter
        hourly_activity = {hour: 0 for hour in range(24)}
        
        # Count conversations by hour
        for convo in conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo:
                    timestamp = datetime.fromisoformat(convo["created_at"].replace('Z', '+00:00'))
                    hour = timestamp.hour
                    hourly_activity[hour] += 1
            except Exception:
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
            
            bars = ax.bar(hours, activity_counts, color='#3498DB', alpha=0.7, edgecolor='white', linewidth=1)
            
            # Highlight peak hour
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
        save_plot_to_supabase(plt, "Peak hours activity plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating peak hours activity plot: {e}")
    finally:
        plt.close()


def generate_day_of_week_activity_plot(conversations, chatbot_id, period):
    """Day of week activity analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        # Initialize day-of-week activity counter
        daily_activity = {day: 0 for day in range(7)}  # 0=Monday, 6=Sunday
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Count conversations by day of week
        for convo in conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo:
                    timestamp = datetime.fromisoformat(convo["created_at"].replace('Z', '+00:00'))
                    day_of_week = timestamp.weekday()
                    daily_activity[day_of_week] += 1
                elif "date_of_convo" in convo:
                    # Fallback to date only
                    date_obj = datetime.strptime(convo["date_of_convo"], "%Y-%m-%d")
                    day_of_week = date_obj.weekday()
                    daily_activity[day_of_week] += 1
            except Exception:
                continue

        total_activity = sum(daily_activity.values())
        
        if total_activity == 0:
            ax.text(0.5, 0.5, 'No activity data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            day_counts = [daily_activity[day] for day in range(7)]
            day_bars = ax.bar(day_names, day_counts, color='#27AE60', alpha=0.7, edgecolor='white', linewidth=1)
            
            # Highlight busiest day
            if max(day_counts) > 0:
                busiest_day_idx = day_counts.index(max(day_counts))
                day_bars[busiest_day_idx].set_color('#E74C3C')
                day_bars[busiest_day_idx].set_alpha(0.9)
            
            ax.set_ylabel('Number of Conversations', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add value labels
            for bar, count in zip(day_bars, day_counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(day_counts)*0.02,
                            str(count), ha='center', va='bottom', fontweight='bold')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Day of Week Activity {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Day of week activity plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating day of week activity plot: {e}")
    finally:
        plt.close()


def generate_business_hours_analysis_plot(conversations, chatbot_id, period):
    """Business vs after hours activity analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        # Initialize hourly activity counter
        hourly_activity = {hour: 0 for hour in range(24)}
        
        # Count conversations by hour
        for convo in conversations:
            if "messages" not in convo:
                continue

            try:
                if "created_at" in convo:
                    timestamp = datetime.fromisoformat(convo["created_at"].replace('Z', '+00:00'))
                    hour = timestamp.hour
                    hourly_activity[hour] += 1
            except Exception:
                continue

        total_activity = sum(hourly_activity.values())
        
        if total_activity == 0:
            ax.text(0.5, 0.5, 'No activity data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Business Hours Analysis
            business_hours = list(range(9, 17))  # 9 AM to 5 PM
            after_hours = list(range(0, 9)) + list(range(17, 24))
            
            business_activity = sum(hourly_activity[hour] for hour in business_hours)
            after_hours_activity = sum(hourly_activity[hour] for hour in after_hours)
            
            time_periods = ['Business Hours\n(9AM-5PM)', 'After Hours\n(5PM-9AM)']
            time_counts = [business_activity, after_hours_activity]
            time_colors = ['#3498DB', '#E67E22']
            
            if sum(time_counts) > 0:
                wedges, texts, autotexts = ax.pie(time_counts, labels=time_periods, colors=time_colors,
                                                  autopct='%1.1f%%', startangle=90,
                                                  textprops={'fontsize': 12, 'fontweight': 'bold'},
                                                  wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
                
                # Center text
                ax.text(0, 0, f'{sum(time_counts):,}\nTotal', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='#2C3E50')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Business vs After Hours {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Business hours analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating business hours analysis plot: {e}")
    finally:
        plt.close()


def generate_conversation_quality_analysis(conversations, chatbot_id, period):
    """Quality score distribution analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Analyze conversation quality metrics
            quality_scores = []
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                
                if not user_messages:
                    continue

                # Calculate metrics
                total_messages = len(messages)
                user_count = len(user_messages)
                assistant_count = len(assistant_messages)
                response_ratio = assistant_count / user_count if user_count > 0 else 0
                
                # Check for unanswered queries
                has_unanswered = any("Oops" in msg.get("content", "") for msg in assistant_messages)
                
                # Quality score (0-100)
                quality_score = min(100, (
                    (min(total_messages, 20) / 20 * 30) +  # Conversation depth
                    (min(response_ratio, 2) / 2 * 25) +    # Response coverage
                    (30 if not has_unanswered else 0) +    # Resolution success
                    (15 if total_messages > 5 else total_messages * 3)  # Engagement level
                ))
                
                quality_scores.append(quality_score)
            
            if not quality_scores:
                ax.text(0.5, 0.5, 'No valid conversation data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Quality Score Distribution
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
                
                # Add percentage labels
                total_convos = len(quality_scores)
                for bar, count in zip(bars, score_counts):
                    if count > 0:
                        percentage = (count / total_convos) * 100
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(score_counts)*0.02,
                                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                                fontweight='bold', fontsize=10)
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Quality Score Distribution {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Conversation quality analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating conversation quality analysis: {e}")
    finally:
        plt.close()


def generate_quality_correlation_plot(conversations, chatbot_id, period):
    """Message depth vs quality correlation analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            quality_scores = []
            message_depths = []
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                
                if not user_messages:
                    continue

                total_messages = len(messages)
                user_count = len(user_messages)
                assistant_count = len(assistant_messages)
                response_ratio = assistant_count / user_count if user_count > 0 else 0
                has_unanswered = any("Oops" in msg.get("content", "") for msg in assistant_messages)
                
                quality_score = min(100, (
                    (min(total_messages, 20) / 20 * 30) +
                    (min(response_ratio, 2) / 2 * 25) +
                    (30 if not has_unanswered else 0) +
                    (15 if total_messages > 5 else total_messages * 3)
                ))
                
                quality_scores.append(quality_score)
                message_depths.append(total_messages)
            
            if not quality_scores:
                ax.text(0.5, 0.5, 'No valid conversation data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.scatter(message_depths, quality_scores, alpha=0.6, s=60, 
                           c=quality_scores, cmap='RdYlGn', edgecolors='white', linewidth=1)
                
                # Add trend line
                if len(message_depths) > 2:
                    z = np.polyfit(message_depths, quality_scores, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(message_depths), max(message_depths), 100)
                    ax.plot(x_trend, p(x_trend), "--", color='#E74C3C', linewidth=2, alpha=0.8)
                
                ax.set_xlabel('Total Messages per Conversation', fontsize=12, fontweight='bold')
                ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add correlation coefficient
                correlation = np.corrcoef(message_depths, quality_scores)[0, 1] if len(message_depths) > 1 else 0
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", alpha=0.8),
                        fontsize=11, fontweight='bold')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Message Depth vs Quality Correlation {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Quality correlation plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating quality correlation plot: {e}")
    finally:
        plt.close()


def generate_resolution_analysis_plot(conversations, chatbot_id, period):
    """Problem resolution rate analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
            ax.text(0.5, 0.5, 'No conversation data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            resolution_status = []
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                
                if not user_messages:
                    continue

                # Check for unanswered queries
                has_unanswered = any("Oops" in msg.get("content", "") for msg in assistant_messages)
                resolution_status.append("Resolved" if not has_unanswered else "Unresolved")
            
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
                    
                    wedges, texts, autotexts = ax.pie(resolution_data, labels=resolution_labels, 
                                                      colors=resolution_colors, autopct='%1.1f%%',
                                                      startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                                      wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
                    
                    # Center text
                    resolution_rate = (resolved_count / (resolved_count + unresolved_count)) * 100
                    ax.text(0, 0, f'{resolution_rate:.1f}%\nResolution\nRate', ha='center', va='center', 
                            fontsize=14, fontweight='bold', color='#2C3E50')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Problem Resolution Analysis {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Resolution analysis plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating resolution analysis plot: {e}")
    finally:
        plt.close()


def generate_user_engagement_funnel(conversations, chatbot_id, period):
    """User engagement funnel analysis"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
            ax.text(0.5, 0.5, 'No engagement data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Analyze engagement patterns
            engagement_stages = {
                "Initial Contact": 0,
                "Engaged (2-4 msgs)": 0,
                "Active (5-9 msgs)": 0,
                "Highly Engaged (10-19 msgs)": 0,
                "Power Users (20+ msgs)": 0
            }
            
            user_message_counts = []
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                user_count = len(user_messages)
                
                if user_count == 0:
                    continue
                
                user_message_counts.append(user_count)
                
                # Categorize engagement
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
            
            if not user_message_counts:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Engagement Funnel
                stages = list(engagement_stages.keys())
                values = list(engagement_stages.values())
                colors = ['#1ABC9C', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C']
                
                # Create horizontal funnel
                y_positions = np.arange(len(stages))[::-1]  # Reverse for funnel effect
                bars = ax.barh(y_positions, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                
                # Add labels with percentages
                initial_users = values[0] if values else 1
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        percentage = (value / initial_users) * 100
                        ax.text(value + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
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
        save_plot_to_supabase(plt, "User engagement funnel plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating user engagement funnel: {e}")
    finally:
        plt.close()


def generate_user_message_distribution(conversations, chatbot_id, period):
    """User message count distribution"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
            ax.text(0.5, 0.5, 'No engagement data available', 
                   ha='center', va='center', fontsize=16, color='#7F8C8D')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            user_message_counts = []
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                user_count = len(user_messages)
                
                if user_count > 0:
                    user_message_counts.append(user_count)
            
            if not user_message_counts:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.hist(user_message_counts, bins=min(20, max(user_message_counts)), 
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
        save_plot_to_supabase(plt, "User message distribution plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating user message distribution: {e}")
    finally:
        plt.close()


def generate_engagement_level_distribution(conversations, chatbot_id, period):
    """Engagement level pie chart"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        if not conversations:
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
            
            for convo in conversations:
                if "messages" not in convo:
                    continue

                messages = convo["messages"]
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                user_count = len(user_messages)
                
                if user_count == 0:
                    continue
                
                # Categorize engagement
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
            
            # Filter out zero values for pie chart
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
                
                # Center text
                total_users = sum(pie_values)
                ax.text(0, 0, f'{total_users:,}\nUsers', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='#2C3E50')
            else:
                ax.text(0.5, 0.5, 'No valid engagement data', 
                       ha='center', va='center', fontsize=16, color='#7F8C8D')
        
        period_text = f"({period} days)" if period > 0 else "(All Time)"
        ax.set_title(f'Engagement Level Distribution {period_text}', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        save_plot_to_supabase(plt, "Engagement level distribution plot", chatbot_id, period)
        
    except Exception as e:
        print(f"Error generating engagement level distribution: {e}")
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

        # Generate all individual plots (16 total)
        # Message Analysis (4 plots)
        generate_message_distribution(user_queries, assistant_responses, chatbot_id, period)
        generate_message_length_analysis(user_queries, assistant_responses, chatbot_id, period)
        generate_message_complexity_analysis(user_queries, assistant_responses, chatbot_id, period)
        generate_key_performance_metrics(user_queries, assistant_responses, chatbot_id, period)
        
        # Sentiment Analysis (2 plots)
        generate_sentiment_analysis(user_queries, chatbot_id, period)
        generate_sentiment_score_distribution(user_queries, chatbot_id, period)
        
        # Chat Volume (1 plot)
        generate_chat_volume_plot(conversations, chatbot_id, period)
        
        # Peak Hours Activity (3 plots)
        generate_peak_hours_activity_plot(conversations, chatbot_id, period)
        generate_day_of_week_activity_plot(conversations, chatbot_id, period)
        generate_business_hours_analysis_plot(conversations, chatbot_id, period)
        
        # Quality Analysis (3 plots)
        generate_conversation_quality_analysis(conversations, chatbot_id, period)
        generate_quality_correlation_plot(conversations, chatbot_id, period)
        generate_resolution_analysis_plot(conversations, chatbot_id, period)
        
        # User Engagement (3 plots)
        generate_user_engagement_funnel(conversations, chatbot_id, period)
        generate_user_message_distribution(conversations, chatbot_id, period)
        generate_engagement_level_distribution(conversations, chatbot_id, period)

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
