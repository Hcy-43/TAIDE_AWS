import os
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

load_dotenv()

def convert_pdf_to_markdown_with_azure_di(pdf_path):
    # Get Azure Document Intelligence endpoint and key from environment variables
    doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    # Initialize Azure Document Intelligence loader
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=pdf_path,
        api_key=doc_intelligence_key,
        api_endpoint=doc_intelligence_endpoint,
        api_model="prebuilt-layout"
    )
    
    # Load the document and convert to Markdown
    docs = loader.load()
    return docs[0].page_content  # Return the converted Markdown content

def save_markdown(content, output_path):
    # Save the Markdown content to a file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)

def convert_and_save_all_pdfs_in_directory(pdf_directory, md_directory):
    # Ensure the output Markdown directory exists
    os.makedirs(md_directory, exist_ok=True)

    # Iterate over all files in the specified PDF directory
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, file_name)
            
            # Convert the PDF to Markdown
            markdown_content = convert_pdf_to_markdown_with_azure_di(pdf_path)
            
            # Define the output path in the Markdown directory
            markdown_file_name = file_name.replace(".pdf", ".md")
            markdown_output_path = os.path.join(md_directory, markdown_file_name)
            
            # Save the Markdown content
            save_markdown(markdown_content, markdown_output_path)
            print(f"Converted {file_name} to Markdown and saved as {markdown_file_name} in {md_directory}")

# Specify the directories for PDF input and Markdown output
pdf_directory = "../docs/三大癌症_pdf"
md_directory = "../docs/三大癌症_md"

convert_and_save_all_pdfs_in_directory(pdf_directory, md_directory)
