import gradio as gr
import re
import nltk
import pdfplumber
import docx
import textstat
from io import BytesIO
from newspaper import Article
from collections import Counter
from transformers import pipeline

nltk.download('punkt')

# Load summarization models
summarizers = {
    "T5 (t5-small)": pipeline("summarization", model="t5-small"),
    "BART (bart-large-cnn)": pipeline("summarization", model="facebook/bart-large-cnn"),
    "Pegasus (xsum)": pipeline("summarization", model="google/pegasus-xsum")
}

# Load QA models
qa_models = {
    "DistilBERT QA": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
    "BERT QA": pipeline("question-answering", model="deepset/bert-base-cased-squad2")
}

# Utility functions
def extract_text_from_file(file):
    if file is None:
        return ""
    name = file.name
    ext = name.split('.')[-1]
    if ext == 'txt':
        return file.read().decode()
    elif ext == 'pdf':
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif ext == 'docx':
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def fetch_url_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def get_keywords(text, n=5):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    common = Counter(words).most_common(n)
    return "; ".join(word for word, _ in common)

def summarize_text(text, model_name, min_len, max_len, format_type):
    summary_chunks = []
    for i in range(0, len(text), 1024):
        chunk = text[i:i+1024]
        result = summarizers[model_name](chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        summary_chunks.append(result)
    summary = " ".join(summary_chunks)
    if format_type == "Bullet Points":
        bullets = re.split(r'(?<=[.!?]) +', summary)
        return "\n".join(f"• {point}" for point in bullets if point.strip())
    return summary

def qa_answers(text, questions, model_name):
    model = qa_models[model_name]
    answers = []
    for q in questions.split('\n'):
        if q.strip():
            ans = model(question=q, context=text)
            answers.append(f"{q}: {ans['answer']} (score: {ans['score']:.2f})")
    return "\n".join(answers)

def get_metrics(original, summary):
    return {
        'Input Word Count': len(original.split()),
        'Summary Word Count': len(summary.split()),
        'Compression Rate (%)': round(100 - (len(summary.split()) / len(original.split()) * 100), 2) if len(original.split()) else 0,
        'Readability (Flesch)': textstat.flesch_reading_ease(summary) if summary else 0
    }

# Gradio main function
def process_text(input_text, file, url, summarizer_model, qa_model, min_tokens, max_tokens, format_type, questions):
    if file is not None:
        text = extract_text_from_file(file)
    elif url:
        text = fetch_url_text(url)
    else:
        text = input_text

    if not text:
        return "No input provided.", "", "", "", ""

    summary = summarize_text(text, summarizer_model, min_tokens, max_tokens, format_type)
    keywords = get_keywords(text)
    answers = qa_answers(text, questions, qa_model) if questions else "No questions provided."
    metrics = get_metrics(text, summary)

    metrics_str = f"""
    Input Word Count: {metrics['Input Word Count']}
    Summary Word Count: {metrics['Summary Word Count']}
    Compression Rate: {metrics['Compression Rate (%)']}%
    Readability Score (Flesch): {metrics['Readability (Flesch)']}
    """

    return summary, keywords, answers, metrics_str, text

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Advanced Text Summarizer & Q&A App\nUpload text/file/url, summarize, extract keywords, and ask questions.")

    with gr.Row():
        input_text = gr.Textbox(label="Paste Text Here", placeholder="Enter text...", lines=6)
        file = gr.File(label="Upload File (.txt, .pdf, .docx)")
        url = gr.Textbox(label="URL", placeholder="https://...")

    with gr.Row():
        summarizer_model = gr.Dropdown(choices=list(summarizers.keys()), value="BART (bart-large-cnn)", label="Summarizer Model")
        qa_model = gr.Dropdown(choices=list(qa_models.keys()), value="DistilBERT QA", label="QA Model")

    with gr.Row():
        min_tokens = gr.Slider(5, 300, value=30, step=1, label="Min Tokens")
        max_tokens = gr.Slider(50, 1024, value=120, step=1, label="Max Tokens")

    format_type = gr.Radio(choices=['Paragraph', 'Bullet Points'], value='Paragraph', label="Output Format")
    questions = gr.Textbox(label="Questions (one per line)", placeholder="Type questions...", lines=3)

    process_btn = gr.Button("Process")

    summary_out = gr.Textbox(label="Summarized Text", lines=6)
    keywords_out = gr.Textbox(label="Top Keywords")
    answers_out = gr.Textbox(label="QA Answers", lines=4)
    metrics_out = gr.Textbox(label="Metrics")
    original_out = gr.Textbox(label="Original Text", lines=6)

    process_btn.click(
        fn=process_text,
        inputs=[input_text, file, url, summarizer_model, qa_model, min_tokens, max_tokens, format_type, questions],
        outputs=[summary_out, keywords_out, answers_out, metrics_out, original_out]
    )

if __name__ == "__main__":
    demo.launch()