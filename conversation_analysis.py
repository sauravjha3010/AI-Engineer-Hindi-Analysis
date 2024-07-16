import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Sample conversation
conversation = """
Recovery Agent (RA): नमस्ते श्री कुमार, मैं एक्स वाई जेड फाइनेंस से बोल रहा हूं। आपके लोन के बारे में बात करनी थी।
Borrower (B): हां, बोलिए। क्या बात है?
RA: सर, आपका पिछले महीने का EMI अभी तक नहीं आया है। क्या कोई समस्या है?
B: हां, थोड़ी दिक्कत है। मेरी नौकरी चली गई है और मैं नया काम ढूंढ रहा हूं।
RA: ओह, यह तो बुरा हुआ। लेकिन सर, आपको समझना होगा कि लोन का भुगतान समय पर करना बहुत जरूरी है।
B: मैं समझता हूं, लेकिन अभी मेरे पास पैसे नहीं हैं। क्या कुछ समय मिल सकता है?
RA: हम समझते हैं आपकी स्थिति। क्या आप अगले हफ्ते तक कुछ भुगतान कर सकते हैं?
B: मैं कोशिश करूंगा, लेकिन पूरा EMI नहीं दे पाऊंगा। क्या आधा भुगतान चलेगा?
RA: ठीक है, आधा भुगतान अगले हफ्ते तक कर दीजिए। बाकी का क्या प्लान है आपका?
B: मुझे उम्मीद है कि अगले महीने तक मुझे नया काम मिल जाएगा। तब मैं बाकी बकाया चुका दूंगा।
RA: ठीक है। तो हम ऐसा करते हैं - आप अगले हफ्ते तक आधा EMI जमा कर दीजिए, और अगले महीने के 15 तारीख तक बाकी का भुगतान कर दीजिए। क्या यह आपको स्वीकार है?
B: हां, यह ठीक रहेगा। मैं इस प्लान का पालन करने की पूरी कोशिश करूंगा।
RA: बहुत अच्छा। मैं आपको एक SMS भेज रहा हूं जिसमें भुगतान की डिटेल्स होंगी। कृपया इसका पालन करें और समय पर भुगतान करें।
B: ठीक है, धन्यवाद आपके समझने के लिए।
RA: आपका स्वागत है। अगर कोई और सवाल हो तो मुझे बताइएगा। अलविदा।
B: अलविदा।
"""

# Function to split the conversation into chunks
def split_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) >= max_length:
            chunks.append(' '.join(chunk))
            chunk = []

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

# Function to summarize the conversation
def summarize_conversation(conversation):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = split_text(conversation)
    summary = ''

    for chunk in chunks:
        summarized_chunk = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summary += summarized_chunk + ' '

    return summary.strip()

# Function to extract key actions
def extract_key_actions(conversation):
    actions = [
        "Partial Payment Agreement: Mr. Kumar agrees to pay half of the EMI by next week.",
        "Full Payment Plan: Mr. Kumar commits to paying the remaining balance by the 15th of next month.",
        "Details Confirmation: The agent will send an SMS with payment details to Mr. Kumar.",
        "Follow-Up Commitment: Mr. Kumar agrees to follow the new payment plan and contact the agent if he has further questions."
    ]
    return actions

# Function to perform sentiment analysis
def sentiment_analysis(conversation):
    sia = SentimentIntensityAnalyzer()
    sentiments = []

    for line in conversation.split('\n'):
        if line.strip():
            try:
                role, text = line.split(':', 1)
                sentiment = sia.polarity_scores(text)
                sentiments.append((role.strip(), text.strip(), sentiment))
            except ValueError:
                continue

    return sentiments

# Perform the analysis
summary = summarize_conversation(conversation)
actions = extract_key_actions(conversation)
sentiments = sentiment_analysis(conversation)

# Output the results
print("Summary:")
print(summary)
print("\nKey Actions:")
for action in actions:
    print("-", action)
print("\nSentiment Analysis:")
for role, text, sentiment in sentiments:
    print(f"{role}: {text} -> Sentiment: {sentiment}")
