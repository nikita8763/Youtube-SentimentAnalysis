import gradio as gr
from transformers import pipeline, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def chunk_text(text, max_tokens=512):
    """
    Splits the transcript text into smaller chunks of specified token size (max_tokens).
    """
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        if len(chunk) <= max_tokens:
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    
    return chunks

def analyze_transcript_sentiment(video_id):
    """
    Fetches the transcript and performs sentiment analysis on each line.
    Includes the video_id in the output for reference.
    """
    sentiments = []
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])

        lines = transcript_text.splitlines()

        for line in lines:
            if len(tokenizer.encode(line)) <= 512: 
                sentiment = sentiment_analyzer(line)[0]
                sentiments.append({
                    "line": line,
                    "label": sentiment["label"],
                    "score": sentiment["score"]
                })

        return sentiments
    except Exception as e:
        return f"Error: {e}"

def sentiment_analysis_ui(video_id):
    results = analyze_transcript_sentiment(video_id)
    if isinstance(results, str):
        return results

    positive_count = sum(1 for result in results if result['label'] == 'POSITIVE')
    negative_count = sum(1 for result in results if result['label'] == 'NEGATIVE')

    total_count = len(results)
    positive_percentage = (positive_count / total_count * 100) if total_count > 0 else 0
    negative_percentage = (negative_count / total_count * 100) if total_count > 0 else 0

    output = ""
    for result in results:
        output += f"Line: {result['line']}\nSentiment: {result['label']} (Score: {result['score']:.2f})\n\n"

    output += f"Summary:\nPositive Responses: {positive_count} ({positive_percentage:.2f}%)\n"
    output += f"Negative Responses: {negative_count} ({negative_percentage:.2f}%)\n"
    
    return output.strip()

iface = gr.Interface(
    fn=sentiment_analysis_ui,
    inputs=gr.Textbox(label="YouTube Video ID", placeholder="Enter a YouTube video ID"),
    outputs="text",
    title="YouTube Video Sentiment Analysis",
    description="Enter a YouTube video ID to analyze the sentiment of each line by line."
)

iface.launch()
