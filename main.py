import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Data loading and preprocessing placeholder
def load_data():
    text = """
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
    return text.lower().strip()


def preprocess_text(text, seq_length=10):
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    dataX, dataY = [], []
    for i in range(0, len(text) - seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    return np.array(dataX), np.array(dataY), char_to_int, int_to_char

def build_model(vocab_size, seq_length):
    model = Sequential([
        Embedding(vocab_size, 64, input_length=seq_length),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train():
    text = load_data()
    seq_length = 10
    X, y, char_to_int, int_to_char = preprocess_text(text, seq_length)
    vocab_size = len(char_to_int)
    y_cat = to_categorical(y, num_classes=vocab_size)
    model = build_model(vocab_size, seq_length)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y_cat, epochs=80, batch_size=64, callbacks=[early_stop], verbose=1)
    model.save('lstm_textgen.h5')
    print('Model trained and saved.')
    return model, char_to_int, int_to_char, seq_length

def generate_text(model, char_to_int, int_to_char, seq_length, seed_text, n_chars=100):
    result = seed_text
    for _ in range(n_chars):
        x = np.array([[char_to_int.get(c, 0) for c in result[-seq_length:]]])
        pred = model.predict(x, verbose=0)
        next_index = np.argmax(pred)
        next_char = int_to_char[next_index]
        result += next_char
    return result

def generate_content(model, char_to_int, int_to_char, seq_length, seed, content_type, n_chars=200):
    if content_type == 'caption':
        print("Instagram Caption Example:")
        print(generate_text(model, char_to_int, int_to_char, seq_length, seed, n_chars=50))
    elif content_type == 'news':
        print("News Summary Example:")
        print(generate_text(model, char_to_int, int_to_char, seq_length, seed, n_chars=150))
    elif content_type == 'blog':
        print("Blog Article Example:")
        print(generate_text(model, char_to_int, int_to_char, seq_length, seed, n_chars=300))
    else:
        print("Generic Content Example:")
        print(generate_text(model, char_to_int, int_to_char, seq_length, seed, n_chars))

if __name__ == "__main__":
    model, char_to_int, int_to_char, seq_length = train()
    # Example seeds
    generate_content(model, char_to_int, int_to_char, seq_length, "summer vibes", "caption")
    generate_content(model, char_to_int, int_to_char, seq_length, "breaking news:", "news")
    generate_content(model, char_to_int, int_to_char, seq_length, "how to save time as a content creator", "blog")
