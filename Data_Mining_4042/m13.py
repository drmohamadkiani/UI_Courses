import ollama
import time
import fitz
client = ollama.Client(host='http://<IP>:11434')

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF file"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


class CAGSystem:
    def __init__(self, model_name='llama3.1:8b'):
        self.model = model_name
        self.messages = []

    def load_pdf(self, pdf_path):
        """Load entire PDF into cache as system message"""
        print(f"\n[CAG] Loading PDF: {pdf_path}")

        # Extract full text
        text = extract_text_from_pdf(pdf_path)
        print(f"  ✓ Extracted {len(text)} characters")

        # Store as system message - this gets cached by Ollama
        self.messages = [
            {
                'role': 'system',
                'content': f'You have access to this document. Answer questions based only on this document:\n\n{text}'
            }
        ]


    def ask(self, question):
        """Answer question using cached document"""
        start = time.time()

        # Append question to conversation
        messages = self.messages.copy()
        messages.append({'role': 'user', 'content': question})

        # Generate answer - KV cache is reused for the document
        response = client.chat(
            model=self.model,
            messages=messages,
            options={'num_ctx': 8192}  # Adjust based on document size
        )

        answer = response['message']['content']

        # Update conversation history
        self.messages.append({'role': 'user', 'content': question})
        self.messages.append({'role': 'assistant', 'content': answer})

        elapsed = time.time() - start
        return answer, elapsed

cag = CAGSystem()

