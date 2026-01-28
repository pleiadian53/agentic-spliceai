import agentic_doc
import openai

# Load your API keys
LANDINGAI_API_KEY = "your_landingai_api_key"
OPENAI_API_KEY = "your_openai_api_key"
openai.api_key = OPENAI_API_KEY

# URL of the OpenSpliceAI GitHub README.md (Raw Markdown)
source_url = "https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/main/README.md"

# Initialize agentic-doc extractor
extractor = agentic_doc.DocumentExtraction(api_key=LANDINGAI_API_KEY)

# Prompt clearly specifying extraction of methodologies
prompt = "Extract detailed information about methodologies, including AI models, training methods, datasets used, and evaluation metrics from the document."

# Extract content from the URL
extracted_response = extractor.extract_from_url(url=source_url, prompt=prompt)

# Define a follow-up prompt for restructuring and paraphrasing
refactor_prompt = (
    "Take the extracted methodologies below and organize them concisely and clearly. "
    "Paraphrase the descriptions to facilitate systematic follow-up by other AI agents. "
    "Focus on structured logic and clarity, suitable for recreating or refactoring the source code: \n\n"
    f"{extracted_response.text}"
)

# Use OpenAI's GPT model to restructure and paraphrase the extracted content
restructured_response = openai.ChatCompletion.create(
    model="gpt-4-turbo-preview",
    messages=[{"role": "user", "content": refactor_prompt}],
    max_tokens=1000,
    temperature=0.3
)

# Output restructured and paraphrased methodologies
print(restructured_response.choices[0].message.content)
