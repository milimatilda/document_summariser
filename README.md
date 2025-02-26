# PDF Summarization Tool  

A **Streamlit-based** web application for summarizing PDF documents using **extractive** and **abstractive** summarization techniques.  

## Features  

- Extracts text from uploaded PDF files.  
- Supports **extractive summarization** using BERT, ALBERT, and DistilBERT models.  
- Supports **abstractive summarization** using models like BART, Pegasus, and DistilBART.  
- Allows users to select the summarization method and model of choice.  
- Displays the summary along with execution time.  

## Supported Models

### Extractive Summarization:
	•	BERT (bert-large-uncased)
	•	ALBERT (albert-base-v2)
	•	DistilBERT (distilbert-base-uncased)

### Abstractive Summarization:
	•	BART (facebook/bart-large-cnn)
	•	Pegasus (google/pegasus-xsum)
	•	DistilBART (sshleifer/distilbart-cnn-12-6)

## Logging

The application uses logging to track activities and errors. Logs are output to the console during execution.

