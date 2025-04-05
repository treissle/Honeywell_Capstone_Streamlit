from io import BytesIO
import blake3
import sqlite3
import json
import re
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument
from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DocumentProcessor:
    """
    Class to process documents: convert, scan documents for CUI, and export in various formats

    Attributes:
        file (str, BytesIO): File path or BytesIO object
        file_hash (str): BLAKE3 hash of the document
        text_hashes (Dict[str, bool]): Text hashes with CUI classification
        file_CUI_classification (bool): Document CUI classification
        docling_document (DoclingDocument): DoclingDocument object
    """
    def __init__(self, file: Union[str, BytesIO]):
        self.file = file
        self.file_hash = self.calculate_hash(file=file)

        self.conn = sqlite3.connect('document_processor.db')
        self.cursor = self.conn.cursor()

        # Create tables if they don't exist
        self.create_tables()

        # Check if the file hash already exists in the database
        file_result_db = self.query_from_db()
        
        # If found, set file CUI classification and skip Docling conversion
        if file_result_db is not None:
            self.file_CUI_classification = file_result_db
            self.text_hashes = None
            self.docling_document = None
        
        # If not found, convert the document, run classification, and update database
        else:
            self.docling_document = self.convert_document()
            self.file_CUI_classification, self.text_hashes = self.CUI_classification()
            self.insert_into_db()

        # Close database connection
        self.conn.close()
    

    def create_tables(self):
        """Create necessary tables in the database"""

        # File hashes table to store file hash, file CUI classification, list of text hashes in document
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_hashes (
            file_hash TEXT PRIMARY KEY,
            file_CUI_classification BOOLEAN,
            text_hashes TEXT
        )
        ''')

        # Text hashes table to store text hash, text CUI classification
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_hashes (
            text_hash TEXT PRIMARY KEY,
            text_CUI_classification BOOLEAN
        )
        ''')
        
        # Document details table to store file hash, text hash, self ref (used in Docling document), page number, bounding box information
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_details (
            file_hash TEXT,
            text_hash TEXT,
            self_ref TEXT,
            page_no INTEGER,
            bbox TEXT,
            PRIMARY KEY (file_hash, text_hash),
            FOREIGN KEY (file_hash) REFERENCES file_hashes (file_hash),
            FOREIGN KEY (text_hash) REFERENCES text_hashes (text_hash)
        )
        ''')

        self.conn.commit()


    def calculate_hash(self, file: Union[str, BytesIO] = None, text: str = None) -> str:
        """
        Calculate the BLAKE3 hash of a file or text portion

        Parameters:
        - file (Union[str, BytesIO]): Path to the file or a BytesIO object
        - text (str): Text portion from document

        Returns:
        - hash (str): Hexadecimal representation of the BLAKE3 hash
        """
        # Initialize the BLAKE3 hash object
        hash_obj = blake3.blake3()

        if file:
            # Check if the file is a file path or a BytesIO object
            if isinstance(file, str):
                # Open the file in binary mode
                with open(file, 'rb') as file:
                    # Read the file in chunks to avoid memory issues with large files
                    for chunk in iter(lambda: file.read(4096), b''):
                        hash_obj.update(chunk)
            elif isinstance(file, BytesIO):
                # Read the BytesIO object in chunks
                file.seek(0)
                for chunk in iter(lambda: file.read(4096), b''):
                    hash_obj.update(chunk)
            else:
                raise ValueError("File must be a file path or a BytesIO object")
        
        # Calculate hash of text portion
        elif text:
            hash_obj.update(text.encode('utf-8'))
        else:
            raise ValueError("Either file or text (path or BytesIO object) must be provided")

        # Return the hexadecimal representation of the BLAKE3 hash
        return hash_obj.hexdigest()
    

    
    def query_from_db(self, text_hash: str = None) -> Union[bool, None]:
        """
        Query the SQLite database for file or text hash

        Parameters:
        - text_hash (str): Optional; text hash to query. If None, queries the file hash.

        Returns:
        - Union[bool, None]: File or text CUI classification if found, else None
        """
        # Query the file_hashes table for the file hash
        if text_hash is None:
            self.cursor.execute('SELECT file_CUI_classification FROM file_hashes WHERE file_hash = ?', (self.file_hash,))
            file_result = self.cursor.fetchone()
            
            # Return the file CUI classification if found, else None
            if file_result:
                return file_result[0]
            else:
                return None
        
        # Query the text_hashes table for the text hash
        else:
            self.cursor.execute('SELECT text_CUI_classification FROM text_hashes WHERE text_hash = ?', (text_hash,))
            text_result = self.cursor.fetchone()

            # Return the text CUI classification if found, else None
            if text_result:
                return text_result[0]
            else:
                return None



    def convert_document(self) -> DoclingDocument:
        """
        Converts a document to DoclingDocument using Docling

        Returns:
        - DoclingDocument: Converted document
        """
        try:
            converter = DocumentConverter()
            
            # File path string
            if isinstance(self.file, str):
                result = converter.convert(self.file) 
            
            # BytesIO object
            elif isinstance(self.file, BytesIO):
                self.file.seek(0)  # Ensure the file is read from the beginning
                doc_stream = DocumentStream(stream=self.file)
                result = converter.convert(doc_stream)

            return result.document

        except Exception as e:
            raise ValueError(f"Error converting document: {e}")


    def CUI_classification(self) -> Union[bool, Dict[str, bool]]:
        """
        Scans a DoclingDocument for occurrences of specified keywords (scope 1) or classify text using model

        Returns:
            file_CUI_classification (bool): File CUI classification 
            text_hashes (Dict[str, bool]): Dictionary where keys are text hashes and values are classifications
        """
        # keywords = ["CUI", "Controlled Unclassified Information", "FOUO", "OUO"]
        # keyword_patterns = [r'\b{}\b'.format(re.escape(keyword)) for keyword in keywords]

        file_CUI_classification = False
        text_hashes = {} # Key: text hash, Value: text CUI classification

        for text in self.docling_document.texts:
            # Query text hashes table to see if text portion was classified before
            text_hash = self.calculate_hash(text=text.text)
            text_CUI_classification = self.query_from_db(text_hash)

            if text_CUI_classification is None:
                # Classify text portion (keyword scan, scope 1)
                # text_CUI_classification = any(re.search(pattern, text.text, re.IGNORECASE) for pattern in keyword_patterns)

                text_CUI_classification = classify_text(text.text)
            
            text_hashes[text_hash] = text_CUI_classification
            
            # Set file CUI classification to True if text CUI classification is True
            file_CUI_classification |= text_CUI_classification
        
        return file_CUI_classification, text_hashes


    def insert_into_db(self):
        """
        Insert file hash, text hashes, and document details in SQLite database
        """
        try:
            # Collect all data to be inserted
            file_data = (self.file_hash, self.file_CUI_classification, json.dumps(list(self.text_hashes.keys())))
            text_hashes_data = list(self.text_hashes.items())
            document_details_data = []

            for text_hash, text in zip(self.text_hashes.keys(), self.docling_document.texts):
                for prov_item in text.prov:
                    document_details_data.append(
                        (
                            self.file_hash,
                            text_hash,
                            text.self_ref,
                            prov_item.page_no,
                            json.dumps({
                                'l': prov_item.bbox.l,
                                't': prov_item.bbox.t,
                                'r': prov_item.bbox.r,
                                'b': prov_item.bbox.b
                            })
                        )
                    )

            # Insert data into the database
            with self.conn:
                self.cursor.execute(
                    'INSERT INTO file_hashes (file_hash, file_CUI_classification, text_hashes) VALUES (?, ?, ?)',
                    file_data
                )
                self.cursor.executemany(
                    'INSERT OR IGNORE INTO text_hashes (text_hash, text_CUI_classification) VALUES (?, ?)',
                    text_hashes_data
                )
                self.cursor.executemany(
                    'INSERT INTO document_details (file_hash, text_hash, self_ref, page_no, bbox) VALUES (?, ?, ?, ?, ?)',
                    document_details_data
                )

            print(f"Inserted file with hash {self.file_hash} and classification {self.file_CUI_classification} and text hashes {list(self.text_hashes.keys())}")
            print(f"Inserted {len(text_hashes_data)} text hashes.")
            print(f"Inserted {len(document_details_data)} document details.")

        except Exception as e:
            print(f"Error inserting data into the database: {e}")


    def export_docling_document(self, export_format: str) -> str:
        """
        Export a DoclingDocument to the specified format

        Args:
            export_format (str): Format to export to ('markdown', 'text', 'html', 'json')

        Returns:
            str: Document content in the specified format
        """
        export_format = export_format.lower()

        if export_format == 'markdown':
            return self.docling_document.export_to_markdown()
        elif export_format == 'text':
            return self.docling_document.export_to_text()
        elif export_format == 'html':
            return self.docling_document.export_to_html()
        elif export_format == 'json':
            return json.dumps(self.docling_document.export_to_dict())
        else:
            raise ValueError(f"Unsupported export format: {export_format}")


def classify_text(text: str) -> bool:
    """
    Machine learning model to conduct binary text classification
    
    Args:
        text (str): Text portion to be classified

    Returns:
        bool: Text CUI classification
    """
    model_path = 'trained_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # model.eval()

    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return bool(predicted_class)
