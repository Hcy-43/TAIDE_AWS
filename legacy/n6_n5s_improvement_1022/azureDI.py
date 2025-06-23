# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
# from langchain.text_splitter import MarkdownHeaderTextSplitter

# # Load environment variables
# load_dotenv()

# # Set up environment variables
# doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
# doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# # Function to load the PDF and process with Azure Document Intelligence
# def load_and_chunk_pdf_with_azure_di(pdf_path):
#     # Initialize Azure Document Intelligence loader
#     loader = AzureAIDocumentIntelligenceLoader(
#         file_path=pdf_path,
#         api_key=doc_intelligence_key,
#         api_endpoint=doc_intelligence_endpoint,
#         api_model="prebuilt-layout"
#     )
    
#     # Load the document and convert to Markdown
#     docs = loader.load()

#     # Set up the Markdown header splitter to chunk based on headers
#     headers_to_split_on = [
#         ("#", "Header 1"),
#         ("##", "Header 2"),
#         ("###", "Header 3"),
#     ]
#     text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

#     # Convert the loaded document into string and split it into chunks
#     docs_string = docs[0].page_content  # Azure DIでMarkdown化したコンテンツ
#     splits = text_splitter.split_text(docs_string)  # 設定したsplitterで分割を行う

#     # Return the chunked documents
#     return splits

# # Function to display chunk metadata for verification
# def display_chunk_metadata(splits):
#     for split in splits:
#         print(split.metadata)

# # Main function to process and display PDF chunks
# def main():
#     # PDFファイルのパスを設定 (ファイルパスを適切に設定してください)
#     pdf_path = "docs/認識子宮頸癌 _ 臺中榮民總醫院護理衛教.pdf"
    
#     # PDFをロードしてセマンティックチャンキングを行う
#     splits = load_and_chunk_pdf_with_azure_di(pdf_path)
    
#     # 分割されたコンテンツのメタデータを表示
#     display_chunk_metadata(splits)

# # Run the main function
# if __name__ == "__main__":
#     main()


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Load environment variables
load_dotenv()

# Set up environment variables
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Function to load the PDF and process with Azure Document Intelligence
def load_and_chunk_pdf_with_azure_di(pdf_path):
    # Initialize Azure Document Intelligence loader
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=pdf_path,
        api_key=doc_intelligence_key,
        api_endpoint=doc_intelligence_endpoint,
        api_model="prebuilt-layout"
    )
    
    # Load the document and convert to Markdown
    docs = loader.load()

    # Set up the Markdown header splitter to chunk based on headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Convert the loaded document into string and split it into chunks
    docs_string = docs[0].page_content  # Azure DIでMarkdown化したコンテンツ
    splits = text_splitter.split_text(docs_string)  # 設定したsplitterで分割を行う

    return splits

# Function to display each chunk with its content and metadata
def display_chunks_with_content(splits):
    for i, split in enumerate(splits):
        print(f"--- Chunk {i + 1} ---")
        print(f"Metadata: {split.metadata}")
        print(f"Content:\n{split.page_content}\n")
        print("------------------------")

# Main function to process and display PDF chunks
def main():
    # PDFファイルのパスを設定 (ファイルパスを適切に設定してください)
    # pdf_path = "docs/認識子宮頸癌 _ 臺中榮民總醫院護理衛教.pdf"
    pdf_path = "ped026_test/PED-026全人醫療照顧手冊_table.pdf"
    
    # PDFをロードしてセマンティックチャンキングを行う
    splits = load_and_chunk_pdf_with_azure_di(pdf_path)
    
    # 分割された各チャンクを表示
    display_chunks_with_content(splits)

# Run the main function
if __name__ == "__main__":
    main()
