# Hindi Conversation Analysis

This project analyzes a conversation in Hindi between a recovery agent and a borrower regarding a default payment. The analysis includes summarization, action item extraction, and sentiment analysis.

## Features

- Summarizes the conversation using BART-large-CNN model
- Extracts key actions from the conversation
- Performs sentiment analysis using NLTK's VADER sentiment analyzer

## Requirements

- Python 3.7+
- NLTK
- Transformers
- PyTorch

## Installation

1. Clone this repository:
2. Install the required packages:
3. Download the necessary NLTK data:
import nltk
nltk.download('vader_lexicon')
python conversation_analysis.py

#The script will output:
A summary of the conversation
Key actions extracted from the conversation
Sentiment analysis for each line of dialogue

Note
This project is part of an AI Engineering assignment focused on natural language processing tasks in Hindi. The code demonstrates skills in text summarization, information extraction, and sentiment analysis.
