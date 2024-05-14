import magic
import time
import os
from pymilvus import connections, Collection
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from api_service.controllers.FileController import FileController
import torch.nn.functional as F
from torch import Tensor
import textract
from transformers import AutoTokenizer, AutoModel

class EmbeddingsController:
    def __init__(self):
        print(f" Connection host {os.getenv('MILVUS_HOST')}")
        connections.connect(
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT")
        )
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        self.embedding_dim = 768
        self.collection = None
        self.max_length = 8192
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
        self.create_index_for_collection()

    def create_collection_in_milvus(self, dim=1024):
        if self.collection_name in utility.list_collections():
            print(f"Collection '{self.collection_name}' already exists.")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name='key', dtype=DataType.VARCHAR, is_primary=False, auto_id=False, max_length=256),
                FieldSchema(name='chunk_text', dtype=DataType.VARCHAR, is_primary=False, auto_id=False, max_length=self.max_length + 500)  # Change data type to VARCHAR
            ]
            schema = CollectionSchema(fields=fields, description="Embeddings")
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")
            self.collection = Collection(name=self.collection_name)
        return self.collection

    def create_index_for_collection(self, field_name="embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 1024}):
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": params}

        if self.collection is None:
            self.collection = Collection(name=self.collection_name)

        print(f" collection: {self.collection}")

        if self.collection.has_index():
            self.collection.release()
            self.collection.drop_index()

        self.collection.create_index(field_name=field_name, index_params=index_params)
        print(f"Index created for field '{field_name}' in collection '{self.collection.name}'.")
        self.collection.load() 

    def wait_for_index_building_complete(self):
        print("Waiting for index building to complete...")
        while True:
            status = utility.index_building_progress(self.collection_name)
            indexed_rows = status['indexed_rows']
            total_rows = status['total_rows']
            print(f"Index building progress: {indexed_rows}/{total_rows}")
            if indexed_rows >= total_rows:
                print("Index building complete.")
                break
            time.sleep(2)

    def download_workspace_and_create_embeddings(self, workspace_folder, key):
        # Download the workspace folder
        fileControls = FileController()
        workspace_path = fileControls.download_code_workspace_folder(workspace_folder, key)
        # Create embeddings
        self.create_embeddings_from_workspace(workspace_path, key)
        return {"response": "Success"}

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def generate_bert_embedding(self, text):
        # Truncate the input text to the maximum sequence length
        # input_text = f'{text}'[:max_length]
        input_text = f'{text}'
        
        batch_dict = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze().tolist()

    def create_embeddings_from_workspace(self, workspace_path, key):
        # Create collection
        self.create_collection_in_milvus(dim=self.embedding_dim)
        # Create embeddings
        for root, _, files in os.walk(workspace_path):
            for file in files:
                file_path = os.path.join(root, file)
                text = self.extract_text_from_file(file_path)
                if text:
                    embeddings = []
                    keys = []
                    chunk_texts = []
                    chunks = self.chunk_text(text)
                    for i, chunk in enumerate(chunks, start=1):
                        print(f"Processing chunk {i}/{len(chunks)} from {file_path}...")
                        embedding = self.generate_bert_embedding(chunk)
                        embeddings.append(embedding)
                        keys.append(f"{key}_{file_path}_{i}")  # Include file path in the key
                        chunk_texts.append(chunk)

                    data = [{"embedding": emb, "key": k, "chunk_text": text} for emb, k, text in zip(embeddings, keys, chunk_texts)]
                    self.insert_embeddings_to_milvus(data)
        return {"response": "Success"}

    def split_long_chunk(self, chunk):
        max_length = self.max_length
        split_chunks = []
        while len(chunk) > max_length:
            split_chunks.append(chunk[:max_length])
            chunk = chunk[max_length:]
        if chunk:
            split_chunks.append(chunk)
        return split_chunks

    def extract_text_from_file(self, file_path):
        try:
            # Define a dictionary of coding language extensions and frameworks
            coding_extensions = {
                '.git': 'Git',
                '.gitignore': 'GitIgnore',
                
                '.py': 'Python',
                '.r': 'R',
                '.cs': 'C#',
                '.cpp': 'C++',
                '.c': 'C',
                '.java': 'Java',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.jsx': 'JSX',
                '.tsx': 'TSX',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.go': 'Go',
                '.swift': 'Swift',
                '.kt': 'Kotlin',
                '.scala': 'Scala',
                '.rs': 'Rust',
                '.sh': 'Shell',
                '.bat': 'Batch',
                '.ps1': 'PowerShell',
                '.sql': 'SQL',
                '.html': 'HTML',
                '.css': 'CSS',
                '.sass': 'Sass',
                '.scss': 'Scss',
                '.less': 'Less',
                '.xml': 'XML',
                '.json': 'JSON',
                '.yml': 'YAML',
                '.yaml': 'YAML',
                '.ini': 'INI',
                '.toml': 'TOML',
                '.md': 'Markdown',
                '.rst': 'reStructuredText',
                '.txt': 'Plain Text',
                '.vue': 'Vue',
                '.svelte': 'Svelte',
                '.dart': 'Dart',
                '.dockerfile': 'Dockerfile',
                '.graphql': 'GraphQL',
                '.proto': 'Protocol Buffers',
                '.pl': 'Perl',
                '.lua': 'Lua',
                '.clj': 'Clojure',
                '.erl': 'Erlang',
                '.hs': 'Haskell',
                '.ml': 'OCaml',
                '.f90': 'Fortran',
                '.asm': 'Assembly',
                '.vba': 'VBA',
                '.cbl': 'COBOL',
                '.pas': 'Pascal',
                '.groovy': 'Groovy',
                '.jsp': 'JSP',
                '.asp': 'ASP',
                '.aspx': 'ASP.NET',
                '.cshtml': 'Razor',
                '.vb': 'Visual Basic',
                '.fs': 'F#',
                '.lsp': 'Lisp',
                '.prolog': 'Prolog',
                '.m': 'MATLAB',
                '.r': 'R',
                '.sas': 'SAS',
                '.jl': 'Julia',
                '.coffee': 'CoffeeScript',
                '.ts': 'TypeScript',
                '.dart': 'Dart',
                '.kt': 'Kotlin',
                '.kts': 'Kotlin Script',
                '.swift': 'Swift',
                '.rs': 'Rust',
                '.go': 'Go',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.erb': 'ERB',
                '.haml': 'Haml',
                '.slim': 'Slim',
                '.ejs': 'EJS',
                '.hbs': 'Handlebars',
                '.pug': 'Pug',
                '.jade': 'Jade',
                '.jinja': 'Jinja',
                '.njk': 'Nunjucks',
                '.twig': 'Twig',
                '.liquid': 'Liquid',
                '.mustache': 'Mustache',
                '.html': 'HTML',
                '.htm': 'HTML',
                '.xhtml': 'XHTML',
                '.css': 'CSS',
                '.sass': 'Sass',
                '.scss': 'Scss',
                '.less': 'Less',
                '.stylus': 'Stylus',
                '.postcss': 'PostCSS',
                '.js': 'JavaScript',
                '.mjs': 'JavaScript Module',
                '.cjs': 'CommonJS',
                '.jsx': 'JSX',
                '.tsx': 'TSX',
                '.json': 'JSON',
                '.webmanifest': 'Web App Manifest',
                '.xml': 'XML',
                '.svg': 'SVG',
                '.mathml': 'MathML',
                '.yml': 'YAML',
                '.yaml': 'YAML',
                '.toml': 'TOML',
                '.ini': 'INI',
                '.md': 'Markdown',
                '.mdx': 'MDX',
                '.rst': 'reStructuredText',
                '.txt': 'Plain Text',
                '.tex': 'LaTeX',
                '.bib': 'BibTeX',
                '.org': 'Org Mode',
                '.wiki': 'Wiki Markup',
                '.asciidoc': 'AsciiDoc',
                '.adoc': 'AsciiDoc',
                '.rdoc': 'RDoc',
                '.pod': 'Pod',
                '.csv': 'CSV',
                '.tsv': 'TSV',
                '.sql': 'SQL',
                '.mysql': 'MySQL',
                '.pgsql': 'PostgreSQL',
                '.plsql': 'PL/SQL',
                '.hql': 'HiveQL',
                '.cql': 'Cassandra Query Language',
                '.ddl': 'Data Definition Language',
                '.dml': 'Data Manipulation Language',
                '.ql': 'GraphQL',
                '.cypher': 'Cypher',
                '.gql': 'GraphQL',
                '.sparql': 'SPARQL',
                '.owl': 'Web Ontology Language',
                '.shex': 'ShEx',
                '.n3': 'Notation3',
                '.ttl': 'Turtle'
            }

            other_extensions = {
                '.csv': 'CSV', 
                '.doc': 'Docs', 
                '.docx': 'Docxs', 
                '.eml': 'Eml', 
                '.epub': 'Epub', 
                '.gif': 'GIF', 
                '.htm': 'htm', 
                '.html':'HTML',
                '.jpeg':'JPEG',
                '.jpg': 'JPEG2',
                '.json': 'JSON',
                '.log': 'LOG',
                '.mp3': 'MP3',
                '.msg': 'MSG',
                '.odt': 'ODT',
                '.ogg': 'OGG',
                '.pdf': 'PDF',
                '.png': 'PNG',
                '.pptx': 'PPTX',
                '.ps': 'PS',
                '.psv': 'PSV',
                '.rtf': 'RTF',
                '.tab': 'TAB',
                '.tff': 'TFF',
                '.tif': 'TIF',
                '.tiff': 'TIFF',
                '.tsv': 'TSV',
                '.txt': 'TXT',
                '.wav': 'WAV',
                '.xls': 'XLS',
                '.xlsx': 'XLSX'
            }
            # Get the file extension
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # Check if the file extension is in the coding_extensions dictionary
            if extension in coding_extensions:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = f"{file_path}\n{file.read()}"
                print(f"Coding language/framework: {coding_extensions[extension]}")
            else:
                # Determine the file type using magic
                file_type = magic.from_file(file_path, mime=True)
                print(f"File type: {file_type}")

                # Extract text based on the file type
                if file_type == 'text/plain':
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = f"{file_path}\n{file.read()}"
                elif extension in other_extensions:
                   text = f"{file_path}\n{textract.process(file_path).decode('utf-8')}"

                elif file_type.startswith('text/'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = f"{file_path}\n{file.read()}"
                else:
                    # Use textract for non-text files
                    text = f"{file_path}\n{textract.process(file_path).decode('utf-8')}"

            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return ""
        
    def chunk_text(self, text):
        max_length = self.max_length - 2  # Accounting for special tokens
        chunks = []
        start = 0
        end = max_length

        while start < len(text):
            # Find the last space within the max_length range
            last_space = text.rfind(" ", start, end)
            if last_space == -1:
                # If no space is found, split at max_length
                chunks.append(text[start:end])
                start = end
                end += max_length
            else:
                # Split at the last space
                chunks.append(text[start:last_space])
                start = last_space + 1
                end = start + max_length

        if start < len(text):
            chunks.append(text[start:])

        return chunks

    def insert_embeddings_to_milvus(self, data):

        try:
            mr = self.collection.insert(data)
            self.collection.load()
            return mr.primary_keys
        except Exception as e:
            print(f"Error inserting data into Milvus: {str(e)}")
            raise
    
    def check_if_embedding_exists(self, key):
        try:
            result = self.collection.query(expr=f"key == '{key}'", output_fields=["key"], limit=1)
            return len(result) > 0
        except Exception as e:
            print(f"Error checking if embedding exists for key {key}: {str(e)}")
            return False

    def embedding_retrieval(self, text, key):
        if self.collection is None:
            self.collection = Collection(name=self.collection_name)
            return

        question_embedding = self.generate_bert_embedding(text)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        top_k = 4
        output_fields = ["key", "chunk_text"]

        results = self.collection.search(
            data=[question_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
            search_params={"params": {"nprobe": 10}, "anns_field": "embedding", "limit": top_k, "output_fields": output_fields, "params": {"nprobe": 10}, "filters": [{"key": key}]}
        )

        compiled_result = ""

        for result in results:
            for r in result:
                chunk_text = r.entity.get("chunk_text")
                compiled_result += chunk_text

        return compiled_result