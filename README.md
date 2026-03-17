# Auto-Content Generation for Digital Marketing

This application uses an LSTM neural network to automatically generate blogs, articles, and captions. It's designed to help digital marketers and content creators save time by producing creative content for platforms like Instagram, news sites, and blogs.

## Features
- Auto-generates Instagram captions, news summaries, and blog articles
- Useful for digital marketing and content creation
- Saves time for content creators
- Easy to extend with your own dataset

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script to train and generate content:
   ```
   python main.py
   ```

## Customization
- Edit the `load_data()` function in `main.py` to use your own text data for training.
- Change the seed text and content type in `main.py` to generate different kinds of content.

## Output
- The trained model is saved as `lstm_textgen.h5`.
- Example generated content is printed for Instagram captions, news summaries, and blog articles.

## Example Usage
```
Instagram Caption Example:
summer vibes

News Summary Example:
breaking news:

Blog Article Example:
how to save time as a content creator
```
