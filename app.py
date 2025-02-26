import logging.config
from pypdf import PdfReader
import streamlit as st
import logging
import time

import os
# Fix for potential library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from summarizer import Summarizer
from transformers import pipeline, AlbertModel, AlbertTokenizer, DistilBertModel, DistilBertTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_text(pdf_file):
    """
    Extracts text from the pdf file

    Parameters:
    pdf_file (UploadedFile): A PDF file uploaded via Streamlit

    Returns:
    text(str): Extracted text from the PDF file

    """
    text = ""
     # Initialize PDF reader
    reader = PdfReader(pdf_file)
    logging.info("Reading text")
    # Iterate through each page in the PDF
    for i in range(0,len(reader.pages)):
        page = reader.pages[i] # Get the page
        text += page.extract_text() # Extract and append text
    return text # Return the extracted text

def extractive_summary(text,model_choice):
    """
    Generates an extractive summary of the provided text based on model_choice.

    Parameters:
    text (str): The input text to summarize.
    model_choice (str): The selected model for extractive summarization.

    Returns:
    str: Extractive summary of the input text.
    """
    if model_choice=='bert-large-uncase':
        summarizer = Summarizer(model_choice)
    elif model_choice=='albert-base-v2':
        albert_model = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)
        albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        summarizer = Summarizer(custom_model=albert_model, custom_tokenizer=albert_tokenizer, random_state = 7)
    elif model_choice=='distilbert':
        distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        summarizer = Summarizer(custom_model=distilbert_model, custom_tokenizer=distilbert_tokenizer, random_state = 7)
    return summarizer(text,min_length=60, ratio=0.01) # Generate summary

def abstractive_summary(text,model_choice):
    """
    Generates an abstractive summary of the provided text.

    Parameters:
    text (str): The input text to summarize.
    model_choice (str): The selected model for abstractive summarization.

    Returns:
    str: Abstractive summary of the input text.
    """
    summarizer = pipeline('summarization',model=model_choice)  # Load the selected model
    output = summarizer(text, max_length=130, min_length=30, do_sample=False) # Generate summary
    return output[0]['summary_text'] # Extract the summarized text from output


def generate_summary(text, summarization_type = 'Abstractive', model_choice='bert-large-uncased'):
    """
    Determines the type of summarization to apply and calls the corresponding function.

    Parameters:
    text (str): The input text to summarize.
    summarization_type (str): Either "Abstractive" or "Extractive" (default: Abstractive).
    model_choice (str): The selected model for summarization.

    Returns:
    str: The generated summary.
    """
    if summarization_type == 'Abstractive':
        return abstractive_summary(text,model_choice)
    else:
        return extractive_summary(text,model_choice)


def main():
    st.title("PDF Summarization tool")
    logging.info("App started")

    # PDF file upload
    uploaded_file = st.file_uploader("Attach the document to summarize", type='pdf')
    logging.info(f"Uploaded file:{uploaded_file}")
    
    # Checks if file is uploaded
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text(uploaded_file)
        logging.info(f"Text:{text}")
        # Proceed only if text extraction was successful
        if text:
            # Radio button to select summarization type
            summarization_type = st.radio("Choose Summarization Method:", ("Extractive", "Abstractive"))

            # Model selection dropdown based on summarization type
            if summarization_type == 'Extractive':
                model_choice = st.selectbox("Choose a model",['None','bert-large-uncased','albert-base-v2','distilbert'])
            else:
                model_choice = st.selectbox("Choose a model",['None','facebook/bart-large-cnn','google/pegasus-xsum','Falconsai/text_summarization','sshleifer/distilbart-cnn-12-6'])
            logging.info(f"Model:{model_choice}")

            # Button to generate summary
            if st.button("View Summary"):
                start_time = time.time() 
                # Ensure a valid model is selected
                if model_choice != 'None':
                    with st.spinner("Summarizing..."):
                        summary = generate_summary(text,summarization_type,model_choice)
                        duration = time.time() - start_time # Calculate processing time
                        logging.info(f"Summary:{summary}")
                        logging.info(f"Duration:{duration:.2f}s")

                        # Store summary in session state
                        st.session_state["summary"] = summary  
                        st.success("Summary loaded successfully! Check below.")

                        # Display the summary with execution time
                        if "summary" in st.session_state:
                            st.text_area("Summary:", f"{st.session_state['summary']}\n\nDuration:{duration:.2f}s", height=300)
                else:
                    st.error("Select a model")


# Run the Streamlit app
if __name__ =='__main__':
    main()

