# Resumeme - Web Content Summarizer

**Resumeme** is an advanced Python tool that extracts content from multiple URLs, cleans HTML, and uses multiple AI APIs (OpenAI, Anthropic, OpenRouter, Ollama) with a fallback system to generate summaries and analysis of web content.

## Main Features

- **Multiple URL extraction**: Processes several web pages simultaneously
- **Smart HTML cleaning**: Configurable to keep or remove specific tags
- **AI fallback system**: Tries multiple AI APIs in sequence until a successful response is obtained
- **Output formats**: JSON or plain text with customizable variables
- **Multiple AI providers**: Support for OpenAI, Anthropic, OpenRouter, and Ollama
- **Prompt templates**: Different prompt styles for various types of analysis
- **Complete logging**: Detailed process logging for debugging

## Installation

### Requirements
- Python 3.7 or higher
- Internet connection for AI APIs

### Installing Dependencies

```bash
# Install main dependencies
pip install requests beautifulsoup4 lxml openai anthropic

# Or use requirements.txt
pip install -r requirements.txt
```

**requirements.txt file:**
```txt
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
openai>=1.0.0
anthropic>=0.7.0
urllib3>=2.0.0
python-dotenv>=1.0.0  # Optional: for environment variables
```

## Configuration

### Basic Structure of config.json

```json
{
  "urls": ["https://example.com"],
  "output": { ... },
  "scraping": { ... },
  "html_cleaning": { ... },
  "ai_api": [ ... ],
  "prompt_templates": { ... },
  "selected_prompt": "default",
  "processing": { ... },
  "logging": { ... }
}
```

### Detailed Explanation of All config.json Options

#### 1. urls (required)

```json
"urls": [
  "https://python.org",
  "https://github.com",
  "https://stackoverflow.com"
]
```

- **Type**: Array of strings
- **Description**: List of URLs to process
- **Required**: Yes

#### 2. output (required)

```json
"output": {
  "file": "results/output.json",
  "type": "json",
  "fields": {
    "title": "Content Summary",
    "content": "$IAresult",
    "source": "$source_url",
    "timestamp": "$timestamp"
  },
  "json_indent": 2,
  "encoding": "utf-8"
}
```

- **file**: Output file path (empty = console output)
- **type**: "json" or "text"/"txt"
- **fields**: Output fields with literals or variables
- **json_indent**: Indentation for JSON (0-4)
- **encoding**: File encoding (utf-8, latin-1, etc.)

#### 3. scraping (optional)

```json
"scraping": {
  "user_agent": "Mozilla/5.0...",
  "timeout": 30,
  "delay_between_requests": 1,
  "max_retries": 3,
  "verify_ssl": true,
  "proxies": {},
  "headers": {}
}
```

- **user_agent**: User agent for HTTP requests
- **timeout**: Maximum wait time in seconds
- **delay_between_requests**: Delay between requests (anti-blocking)
- **max_retries**: Reconnection attempts
- **verify_ssl**: Verify SSL certificates
- **proxies**: HTTP proxies (e.g., `{"http": "...", "https": "..."}`)
- **headers**: Custom HTTP headers

#### 4. html_cleaning (optional)

```json
"html_cleaning": {
  "enabled": true,
  "remove_tags": ["script", "style", "meta"],
  "keep_tags": ["p", "h1", "h2", "h3", "article"],
  "remove_attributes": true,
  "remove_comments": true,
  "extract_text_only": false,
  "min_text_length": 50,
  "clean_whitespace": true
}
```

- **enabled**: Enable/disable cleaning
- **remove_tags**: HTML tags to completely remove
- **keep_tags**: Tags to keep (if empty, all are kept)
- **remove_attributes**: Remove attributes from tags
- **remove_comments**: Remove HTML comments
- **extract_text_only**: Extract only text (without HTML)
- **min_text_length**: Minimum text length to keep
- **clean_whitespace**: Clean multiple whitespace

#### 5. ai_api (required for AI processing)

```json
"ai_api": [
  {
    "name": "OpenAI GPT-4",
    "provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 1500
  },
  {
    "name": "Claude 3",
    "provider": "anthropic",
    "api_key": "sk-ant-...",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.7,
    "max_tokens": 1500
  }
]
```

- **name**: Identifying name for the endpoint
- **provider**: "openai", "anthropic", "openrouter", "ollama"
- **api_key**: API key (not required for local Ollama)
- **model**: Model to use
- **endpoint**: Custom URL (optional)
- **temperature**: Randomness (0-2, higher = more creative)
- **max_tokens**: Maximum tokens in response
- **timeout**: Specific timeout for this API
- **system_prompt**: System prompt (customizable)

#### 6. prompt_templates (optional)

```json
"prompt_templates": {
  "default": "Summarize the following content:\n\n$content\n\nSummary:",
  "detailed": "Analyze this content:\n1. Main topic\n2. Key points\n3. Conclusion\n\nContent:\n$content\n\nAnalysis:",
  "keywords": "Extract 10 keywords:\n\n$content\n\nKeywords:"
}
```

- Customizable prompt templates
- `$content` will be replaced by extracted content
- You can add as many templates as needed

#### 7. selected_prompt (optional)

```json
"selected_prompt": "detailed"
```

- Name of the template to use (must exist in prompt_templates)
- Can be overridden via command line

#### 8. processing (optional)

```json
"processing": {
  "concat_strategy": "sequential",
  "max_total_length": 10000,
  "chunk_size": 3000,
  "retry_delay": 2,
  "language": "auto"
}
```

- **concat_strategy**: "sequential" (all together) or "chunked" (in blocks)
- **max_total_length**: Maximum length of concatenated content
- **chunk_size**: Size of each chunk (if using chunked strategy)
- **retry_delay**: Seconds between API retries
- **language**: Language for processing ("auto" for auto-detection)

#### 9. logging (optional)

```json
"logging": {
  "enabled": true,
  "level": "INFO",
  "file": "resumeme.log",
  "max_size_mb": 10,
  "backup_count": 3
}
```

- **enabled**: Enable logging
- **level**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **file**: Log file
- **max_size_mb**: Maximum log size before rotation
- **backup_count**: Number of backups to keep

### Available Variables in Output Fields

| Variable | Description | Example |
|----------|-------------|---------|
| `$IAresult` | Complete AI result | Summary text |
| `$source_url` | Processed URLs | "https://python.org; https://github.com" |
| `$timestamp` | Processing date and time | "2024-01-15T10:30:00" |
| `$word_count` | Word count of clean content | 1542 |
| `$summary` | Short preview of AI result | "Python is a programming language..." |
| `$title` | Extracted title from HTML | "Welcome to Python.org" |
| `$raw_content` | Original content (first 1000 chars) | "<html>...uncleaned content..." |
| `$clean_content` | Clean content (first 1000 chars) | "Clean text without HTML..." |
| `$used_provider` | AI provider used | "OpenAI GPT-4" |
| `$model` | Specific model used | "gpt-4-turbo-preview" |
| `$total_raw_chars` | Total characters of original content | 12500 |
| `$total_clean_chars` | Total characters of clean content | 9800 |

## Usage

### Basic Execution

```bash
python resumeme.py config.json
```

### Specify Prompt via Command Line

```bash
python resumeme.py config.json detailed
python resumeme.py config.json keywords
python resumeme.py config.json technical
```

### Minimal Configuration Example

```json
{
  "urls": ["https://example.com"],
  "output": {
    "file": "output.json",
    "type": "json",
    "fields": {
      "content": "$IAresult"
    }
  },
  "ai_api": [
    {
      "provider": "openai",
      "api_key": "your-key-here",
      "model": "gpt-3.5-turbo"
    }
  ]
}
```

### Complete Configuration Example with 4 Providers

```json
{
  "urls": ["https://python.org"],
  "output": {
    "file": "result.json",
    "type": "json",
    "fields": {
      "title": "Python.org Analysis",
      "summary": "$IAresult",
      "source": "$source_url",
      "provider": "$used_provider",
      "model": "$model",
      "raw_preview": "$raw_content",
      "clean_preview": "$clean_content"
    }
  },
  "scraping": {
    "user_agent": "Mozilla/5.0",
    "timeout": 30
  },
  "html_cleaning": {
    "enabled": true,
    "remove_tags": ["script", "style"]
  },
  "ai_api": [
    {
      "name": "OpenAI",
      "provider": "openai",
      "api_key": "sk-...",
      "model": "gpt-4-turbo-preview",
      "temperature": 0.7
    },
    {
      "name": "Claude",
      "provider": "anthropic",
      "api_key": "sk-ant-...",
      "model": "claude-3-sonnet",
      "temperature": 0.7
    },
    {
      "name": "OpenRouter",
      "provider": "openrouter",
      "api_key": "sk-or-...",
      "model": "openai/gpt-4",
      "referer": "https://github.com"
    },
    {
      "name": "Ollama",
      "provider": "ollama",
      "model": "llama2",
      "endpoint": "http://localhost:11434/api/chat",
      "timeout": 120
    }
  ],
  "prompt_templates": {
    "default": "Summarize this content:\n\n$content\n\nSummary:",
    "detailed": "Analyze in detail:\n\n$content\n\nAnalysis:"
  },
  "selected_prompt": "default",
  "logging": {
    "enabled": true,
    "level": "INFO"
  }
}
```

## AI Fallback System

The script tries APIs in this order:

1. First provider in the `ai_api` list
2. If it fails, wait `retry_delay` seconds
3. Try with the next provider
4. Continue until a successful response is obtained
5. If all fail, save the error in the output

## API Key Considerations

### OpenAI

- Get your key at: https://platform.openai.com
- Recommended models: `gpt-4-turbo-preview`, `gpt-3.5-turbo`

### Anthropic (Claude)

- Get your key at: https://console.anthropic.com
- Recommended models: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### OpenRouter

- Get your key at: https://openrouter.ai
- Model format: `provider/model` (e.g., `openai/gpt-4`)
- Configure `referer` and `title` for identification

### Ollama

- Local installation: https://ollama.ai
- Useful commands:

```bash
ollama pull llama2
ollama pull mistral
ollama serve  # Start the server
```

- Default endpoint: `http://localhost:11434`

## Troubleshooting

### Error: Cannot resolve imports

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: HTTP request timeout

- Increase `timeout` in scraping
- Check your internet connection
- Consider using `delay_between_requests`

### Error: Invalid API key

- Verify that API keys are correct
- Check usage limits and billing
- For Ollama, verify that the server is running

### Error: Content too long

- Reduce `max_total_length` in processing
- Use smaller `chunk_size`
- Consider processing fewer URLs

## Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is under the MIT License. See the LICENSE file for more details.

## Support

To report bugs or request features, please open an issue in the repository.
