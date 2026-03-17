from flask import Flask, render_template, request
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and mappings
model = load_model('lstm_textgen.h5')
# The mappings must strictly match the unique characters used during training in main.py
training_text = """
content creation is the art of producing valuable and engaging material for audiences online.
bloggers write articles that inform educate and entertain readers on topics they care about.
social media creators share short posts images and videos to connect with their followers daily.
great content tells a compelling story that resonates with the audience and inspires action.
to become a successful content creator you need consistency creativity and a clear vision.
time management is essential for content creators who juggle multiple platforms and deadlines.
plan your content calendar at the start of each week to stay organized and ahead of schedule.
batch your content creation sessions to save time and maintain a steady flow of publications.
use analytics to understand what content performs best and focus on topics your audience loves.
writing a blog post requires research planning drafting editing and publishing in a clear format.
a good headline grabs attention and makes readers want to click and discover more information.
the introduction of a blog post should hook the reader immediately with a strong opening statement.
use subheadings bullet points and short paragraphs to make your content easy to read and scan.
always end your blog posts with a call to action that tells readers exactly what to do next.
instagram captions should be short punchy and relevant to the image or video being shared.
use relevant hashtags to increase the visibility of your instagram posts and reach new audiences.
engage with your followers by responding to comments and messages in a timely and friendly manner.
news summaries should be concise accurate and written in a straightforward journalistic style.
a good news summary captures the who what when where and why of any given news story clearly.
content creators must stay updated with current trends to produce timely and relevant material.
video content is among the most engaging formats online and can significantly boost your reach.
podcasts allow creators to share long form conversations and insights with a dedicated audience.
email newsletters help creators build a direct relationship with their audience outside social media.
search engine optimization helps blog posts rank higher in search results and attract organic traffic.
use keywords naturally throughout your content to improve discoverability without sounding robotic.
personal stories and experiences make content more authentic relatable and emotionally powerful.
collaborating with other creators helps grow your audience and brings fresh perspectives to content.
the best content solves a specific problem or answers a specific question for your target audience.
always proofread your work before publishing to catch errors and ensure a polished final product.
publishing content consistently builds trust with your audience and signals reliability over time.
repurpose your best content across multiple platforms to maximize its reach and impact efficiently.
a strong personal brand helps creators stand out and attract loyal followers in a crowded market.
define your niche and stay focused on it to become a recognized expert in your chosen domain.
content that educates entertains or inspires tends to perform the best across all digital platforms.
writing every day even for just a few minutes helps sharpen your skills and generate new ideas.
the most successful content creators listen to their audience and adapt based on feedback received.
"""
chars = sorted(list(set(training_text.lower())))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
seq_length = 10

def sample_with_temperature(preds, temperature=0.5):
    """Sample from predictions using temperature to add diversity."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(seed_text, content_type, n_chars=200, temperature=0.7):
    # Normalize seed to known vocabulary characters only
    result = ''.join(c if c in char_to_int else ' ' for c in seed_text.lower())
    if not result.strip():
        result = 'hello'
    if content_type == 'caption':
        n_chars = 50
    elif content_type == 'news':
        n_chars = 150
    elif content_type == 'blog':
        n_chars = 300
    # Use temperature sampling instead of argmax to avoid repetition collapse
    for _ in range(n_chars):
        x = np.array([[char_to_int.get(c, 0) for c in result[-seq_length:]]])
        pred = model(x, training=False).numpy()[0]
        next_index = sample_with_temperature(pred, temperature)
        next_char = int_to_char[next_index]
        result += next_char
    # Collapse multiple consecutive spaces into a single space and strip edges
    result = re.sub(r' +', ' ', result).strip()
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    generated = None
    if request.method == 'POST':
        seed = request.form['seed']
        content_type = request.form['type']
        generated = generate_text(seed, content_type)
    return render_template('index.html', generated=generated)

if __name__ == '__main__':
    app.run(debug=True)
