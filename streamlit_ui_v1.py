import io
import os
import json
import boto3
import pdfplumber
import streamlit as st
from datetime import datetime
import prompts  # You'll need to import your prompts module

# Page config and styling
st.set_page_config(
    page_title="Document Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Current date
today = datetime.today().strftime('%Y-%m-%d')

# Initialize session state variables if they don't exist
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def is_scanned_pdf(pdf_file):
    """
    Check if a PDF is a scanned document or contains extractable text
    
    Args:
        pdf_file: BytesIO object containing PDF
        
    Returns:
        bool: True if scanned, False if contains extractable text
    """
    try:
        pdf_file.seek(0)  # Reset file pointer to beginning
        with pdfplumber.open(pdf_file) as pdf:
            if len(pdf.pages) == 0:
                return True  # No pages means we can't extract text
                
            # Check the first 3 pages or all pages if less than 3
            pages_to_check = min(3, len(pdf.pages))
            text_content = ""
            
            for i in range(pages_to_check):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    text_content += text
            
            # If we found at least some text, it's not a scanned PDF
            if len(text_content.strip()) > 100:  # Arbitrary threshold
                return False
            else:
                return True
                
    except Exception as e:
        # If we can't analyze it, assume it's scanned
        st.error(f"Error checking PDF: {str(e)}")
        return True

def insert_document(doc_data):
    """
    Mock function to insert document into database
    In a real app, you would connect to your database here
    
    Args:
        doc_data: Document data to insert
        
    Returns:
        str: Generated object ID
    """
    # In Streamlit, we might just want to return the data
    # or implement actual database integration
    return "doc_" + datetime.now().strftime("%Y%m%d%H%M%S")

def bedrock_calling(system_prompt, prompt_type, pdf_text):
    """
    Call AWS Bedrock API with appropriate prompts
    
    Args:
        system_prompt (str): System prompt to use
        prompt_type (str): Type of prompt to use
        pdf_text (str): Extracted text from PDF
        
    Returns:
        dict: Structured analysis response
    """
    try:
        # AWS Configuration - get from Streamlit secrets or environment
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID") or st.secrets.get("aws_access_key_id", "")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or st.secrets.get("aws_secret_access_key", "")
        
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name='us-east-1'
        )
        
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": f"{system_prompt} \n\n {prompt_type}"
                }
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 250
        }

        bedrock = session.client('bedrock-runtime')
        body = json.dumps(payload)
        model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        response_text = response_body.get("content")[0].get("text")
        
        # Extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        response_text = response_text[start:end]
        return json.loads(response_text)
    
    except Exception as e:
        st.error(f"Error calling Bedrock API: {str(e)}")
        return {"error": str(e)}

def get_document_class(pdf_text):
    """
    Classify document type using classification prompt
    
    Args:
        pdf_text (str): Extracted text from PDF
        
    Returns:
        dict: Document classification details
    """
    with st.status("Classifying document type...", expanded=True) as status:
        classification_prompt = prompts.user_prompt_classification(pdf_text)
        system_prompt = prompts.system_prompt_for_doc_classification
        result = bedrock_calling(system_prompt=system_prompt, prompt_type=classification_prompt, pdf_text=pdf_text)
        status.update(label="Document classification complete!", state="complete", expanded=False)
        return result

def get_document_analysis(doc_class, pdf_text):
    """
    Generate appropriate document analysis based on classification
    
    Args:
        doc_class (int): Document class number
        pdf_text (str): Extracted text from PDF
        
    Returns:
        dict: Document analysis
    """
    with st.status("Generating document analysis...", expanded=True) as status:
        prompt_mapping = {
            0: prompts.user_prompt_poi,
            1: prompts.user_prompt_poa,
            2: prompts.user_prompt_registration,
            3: prompts.user_prompt_ownership,
            4: prompts.user_prompt_tax_return,
            5: prompts.user_prompt_financial
        }

        system_prompt_mapping = {
            0: prompts.system_prompt_for_indentity_doc,
            1: prompts.system_prompt_for_poa_doc,
            2: prompts.system_prompt_for_registration_doc,
            3: prompts.system_prompt_for_ownership_doc,
            4: prompts.system_prompt_for_tax_return_doc,
            5: prompts.system_prompt_for_financial_doc
        }

        if doc_class not in prompt_mapping:
            st.error(f"Invalid document class: {doc_class}")
            return {"error": f"Invalid document class: {doc_class}"}

        user_prompt = prompt_mapping[doc_class](pdf_text, today)
        system_prompt = system_prompt_mapping[doc_class]
        
        result = bedrock_calling(system_prompt=system_prompt, prompt_type=user_prompt, pdf_text=pdf_text)
        status.update(label="Document analysis complete!", state="complete", expanded=False)
        return result

def process_document(uploaded_file):
    """
    Process the uploaded PDF document
    
    Args:
        uploaded_file: UploadedFile object from Streamlit
        
    Returns:
        dict: Analysis results
    """
    try:
        st.session_state.processing = True
        
        # Read file bytes
        file_bytes = uploaded_file.getvalue()
        
        # Create BytesIO object
        pdf_file = io.BytesIO(file_bytes)
        
        # Check if scanned
        if is_scanned_pdf(pdf_file):
            st.error("This appears to be a scanned document. The current version does not support scanned PDFs.")
            st.session_state.processing = False
            return None
        
        # Extract text
        pdf_file.seek(0)  # Reset file pointer
        with pdfplumber.open(pdf_file) as pdf:
            extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        # Show text preview
        with st.expander("Preview extracted text", expanded=False):
            st.text_area("Extracted Text", extracted_text, height=200)
        
        # Classify document
        classification_result = get_document_class(extracted_text)
        doc_class = classification_result.get('class')
        
        # Analyze document
        analysis_result = get_document_analysis(doc_class, extracted_text)
        
        # Prepare final response
        final_response = {
            "document_type": classification_result.get('category'),
            "analysis": analysis_result
        }
        
        # Add mock DB ID
        mongo_obj_id = insert_document(final_response)
        final_response["mongo_obj_id"] = mongo_obj_id
        
        st.session_state.processing = False
        return {"result": final_response}
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.session_state.processing = False
        return None

def display_results(results):
    """
    Display the analysis results in a structured format
    
    Args:
        results: Analysis results dictionary
    """
    if not results or "result" not in results:
        return
    
    result = results["result"]
    
    st.markdown('<div class="sub-header">📄 Document Analysis Results</div>', unsafe_allow_html=True)
    
    # Document type
    st.markdown(f"**Document Type:** {result['document_type']}")
    
    # Analysis
    analysis = result["analysis"]
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Summary", "Detailed Analysis"])
    
    with tab1:
        if "summary" in analysis:
            st.markdown("### Key Points")
            st.markdown(analysis["summary"])
        elif "key_points" in analysis:
            st.markdown("### Key Points")
            st.markdown(analysis["key_points"])
        
        if "validity" in analysis:
            st.markdown("### Document Validity")
            st.markdown(f"**Status:** {analysis['validity'].get('status', 'Unknown')}")
            if "confidence" in analysis['validity']:
                st.markdown(f"**Confidence:** {analysis['validity']['confidence']}%")
            if "notes" in analysis['validity']:
                st.markdown(f"**Notes:** {analysis['validity']['notes']}")
    
    with tab2:
        # Display extracted fields
        if "extracted_fields" in analysis:
            st.markdown("### Extracted Fields")
            fields = analysis["extracted_fields"]
            
            if isinstance(fields, dict):
                # Convert to DataFrame for better display
                import pandas as pd
                df = pd.DataFrame(list(fields.items()), columns=["Field", "Value"])
                st.dataframe(df, use_container_width=True)
            elif isinstance(fields, list):
                for item in fields:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            st.markdown(f"**{k}:** {v}")
                    else:
                        st.text(item)
        
        # Display other sections
        for key, value in analysis.items():
            if key not in ["summary", "key_points", "validity", "extracted_fields"]:
                st.markdown(f"### {key.replace('_', ' ').title()}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                st.markdown(f"**{k}:** {v}")
                        else:
                            st.text(item)
                else:
                    st.text(value)
    
    # Download results
    st.download_button(
        label="Download Analysis (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Main app
def main():
    st.markdown('<div class="main-header">Document Analysis Tool</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Upload a PDF document to generate an automated summary and analysis. This tool identifies document type and extracts key information.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool analyzes various types of documents:
        
        - Proof of Identity Documents
        - Proof of Address Documents
        - Registration Documents
        - Ownership Documents
        - Tax Returns
        - Financial Documents
        
        The analysis includes document classification, key information extraction, and validity assessment.
        
        **Note:** Scanned documents are not supported at this time.
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
        
        # Process button
        if st.button("Analyze Document"):
            with st.spinner("Processing document..."):
                st.session_state.results = process_document(uploaded_file)
    
    # Display results if available
    if st.session_state.results:
        display_results(st.session_state.results)
    
    # Display processing message
    if st.session_state.processing:
        st.info("Document analysis in progress...")

if __name__ == "__main__":
    main()
