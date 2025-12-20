#!/usr/bin/env python3
"""
resumeme.py - Web content scraper and AI summarizer with multi-API fallback

This script:
1. Reads multiple URLs from JSON configuration file
2. Fetches and concatenates content from all URLs
3. Cleans HTML tags based on configuration
4. Sends processed content to AI API for summarization/analysis
5. Outputs results in JSON or text format based on configuration
"""

import json
import sys
import re
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI
import anthropic

class ConfigLoader:
    """Handles loading and validation of configuration file"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def load(self) -> Dict[str, Any]:
        """Load and validate configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file '{self.config_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)

        self._validate_config()
        return self.config

    def _validate_config(self):
        """Validate required configuration fields"""
        required_fields = ['urls', 'output']

        for field in required_fields:
            if field not in self.config:
                print(f"Error: Missing required field '{field}' in configuration.")
                sys.exit(1)

        if not isinstance(self.config['urls'], list) or len(self.config['urls']) == 0:
            print("Error: 'urls' must be a non-empty list.")
            sys.exit(1)

class WebScraper:
    """Handles web scraping operations with retry logic and error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.get('scraping', {}).get('max_retries', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': self.config.get('scraping', {}).get('user_agent',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        custom_headers = self.config.get('scraping', {}).get('headers', {})
        if custom_headers:
            session.headers.update(custom_headers)

        return session

    def fetch_url(self, url: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Fetch content from a single URL with error handling"""
        try:
            timeout = self.config.get('scraping', {}).get('timeout', 60)

            response = self.session.get(
                url,
                timeout=timeout,
                verify=self.config.get('scraping', {}).get('verify_ssl', True),
                proxies=self.config.get('scraping', {}).get('proxies', {}),
                stream=True
            )
            response.raise_for_status()

            max_size_bytes = self.config.get('scraping', {}).get('max_content_size', 10 * 1024 * 1024)

            content_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > max_size_bytes:
                    logging.warning("Content from %s exceeds maximum size limit, truncating", url)
                    content_bytes = content_bytes[:max_size_bytes]
                    break

            encoding = response.encoding
            if 'charset' in response.headers.get('content-type', '').lower():
                match = re.search(r'charset=([\w-]+)', response.headers['content-type'].lower())
                if match:
                    encoding = match.group(1)

            content = content_bytes.decode(encoding or 'utf-8', errors='replace')

            metadata = {
                'url': url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'encoding': encoding,
                'size_bytes': len(content_bytes),
                'fetch_time': datetime.now().isoformat(),
                'content_size_kb': len(content_bytes) / 1024
            }

            return content, metadata

        except requests.exceptions.RequestException as e:
            logging.error("Failed to fetch %s: %s", url, e)
            return None, {'url': url, 'error': str(e), 'fetch_time': datetime.now().isoformat()}

    def fetch_all_urls(self) -> List[Tuple[str, Dict]]:
        """Fetch content from all configured URLs"""
        results = []

        for url in self.config['urls']:
            logging.info("Fetching %s", url)
            content, metadata = self.fetch_url(url)
            results.append((content, metadata))

            delay = self.config.get('scraping', {}).get('delay_between_requests', 2)
            if delay > 0:
                time.sleep(delay)

        return results

class HTMLCleaner:
    """Handles HTML cleaning and text extraction based on configuration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('html_cleaning', {})

    def clean_html(self, html_content: str, metadata: Dict) -> Tuple[str, Dict]:
        """Clean HTML content based on configuration"""
        if not html_content or not self.config.get('enabled', True):
            return html_content, metadata

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            if self.config.get('remove_comments', True):
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()

            remove_tags = self.config.get('remove_tags', [
                'script', 'style', 'meta', 'nav', 'footer', 'header',
                'iframe', 'form', 'button', 'input', 'select', 'textarea',
                'aside', 'svg', 'canvas', 'noscript'
            ])
            for tag in remove_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            if self.config.get('remove_attributes', True):
                for tag in soup.find_all(True):
                    tag.attrs = {}

            keep_tags = self.config.get('keep_tags', [
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'article', 'section', 'main', 'div.content',
                'div.article', 'div.post', 'div.main-content',
                'li', 'ul', 'ol', 'blockquote', 'code', 'pre'
            ])
            if keep_tags:
                all_tags = [tag.name for tag in soup.find_all(True) if tag.name not in keep_tags]
                for tag_name in all_tags:
                    for element in soup.find_all(tag_name):
                        element.unwrap()

            if self.config.get('extract_text_only', True):
                cleaned_content = soup.get_text(separator='\n', strip=True)
            else:
                cleaned_content = str(soup)

            if self.config.get('clean_whitespace', True):
                cleaned_content = self._clean_whitespace(cleaned_content)

            original_size = len(html_content)
            cleaned_size = len(cleaned_content)
            compression_ratio = original_size / cleaned_size if cleaned_size > 0 else 1

            metadata['cleaned'] = True
            metadata['original_length'] = original_size
            metadata['cleaned_length'] = cleaned_size
            metadata['compression_ratio'] = round(compression_ratio, 2)
            metadata['title'] = soup.title.string if soup.title else ''

            logging.info("HTML cleaning: %d → %d chars (ratio: %.2f)",
                        original_size, cleaned_size, compression_ratio)

            return cleaned_content, metadata

        except (AttributeError, TypeError, ValueError) as e:
            logging.error("Error cleaning HTML: %s", e)
            return html_content, {**metadata, 'cleaning_error': str(e)}

    def _clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace from text"""
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Trim whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        # Remove empty lines below minimum length
        min_length = self.config.get('min_text_length', 20)  # Reduced from 50
        lines = [line for line in lines if len(line.strip()) >= min_length or not line.strip()]
        return '\n'.join(lines)

class AIProcessor:
    """Handles interactions with multiple AI APIs with fallback system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ai_apis = config.get('ai_api', [])
        self.retry_delay = config.get('processing', {}).get('retry_delay', 2)
        processing_config = config.get('processing', {})
        self.max_prompt_chars = processing_config.get('max_prompt_chars', 120000)
        self.max_response_tokens = processing_config.get('max_response_tokens', 4000)

    def process_content(self, content: str, metadata: Dict) -> Tuple[str, Dict]:
        """Send content to AI APIs with automatic fallback"""
        if not content:
            return "", {**metadata, 'ai_processing_error': 'Empty content'}

        if not self.ai_apis:
            raise ValueError("No AI APIs configured in 'ai_api'")

        prompt_templates = self.config.get('prompt_templates', {})
        selected_prompt = self.config.get('selected_prompt', 'default')
        prompt_template = prompt_templates.get(selected_prompt,
            "Please summarize the following content concisely:\n\n$content\n\nSummary:")

        estimated_tokens = len(content) // 4
        logging.info("Content for AI: %d chars, ~%d tokens", len(content), estimated_tokens)

        if len(content) > self.max_prompt_chars:
            logging.warning("Content too long (%d chars), truncating to %d chars",
                          len(content), self.max_prompt_chars)

        prompt = prompt_template.replace('$content', content)

        for i, api_config in enumerate(self.ai_apis):
            provider = api_config.get('provider', 'openai').lower()
            api_name = api_config.get('name', f'API-{i+1}')

            logging.info("Trying %s (%s)...", api_name, provider)

            try:
                if provider == 'openai':
                    result, ai_metadata = self._call_openai(prompt, api_config)
                elif provider == 'openrouter':
                    result, ai_metadata = self._call_openrouter(prompt, api_config)
                elif provider == 'ollama':
                    result, ai_metadata = self._call_ollama(prompt, api_config)
                elif provider == 'anthropic':
                    result, ai_metadata = self._call_anthropic(prompt, api_config)
                else:
                    result, ai_metadata = self._call_generic_api(prompt, api_config)

                logging.info("Success with %s", api_name)
                return result, {**metadata, 'ai_metadata': ai_metadata, 'used_provider': api_name}

            except Exception as e:  # pylint: disable=broad-except
                error_msg = str(e)
                logging.warning("Failed with %s: %s...", api_name, error_msg[:100])

                if i < len(self.ai_apis) - 1:
                    next_api = self.ai_apis[i + 1].get('name', f'API-{i+2}')
                    logging.info("Waiting %ss before trying %s...", self.retry_delay, next_api)
                    time.sleep(self.retry_delay)
                    continue
                else:
                    error_msg = f"All AI APIs failed. Last error: {e}"
                    logging.error(error_msg)
                    return "", {**metadata, 'ai_processing_error': error_msg}

    def _call_openai(self, prompt: str, api_config: Dict) -> Tuple[str, Dict]:
        """Call OpenAI API using the new SDK"""
        api_key = api_config.get('api_key')
        if not api_key or api_key == 'your-api-key-here':
            raise ValueError("OpenAI API key not configured")

        client_kwargs = {"api_key": api_key}
        if 'endpoint' in api_config:
            client_kwargs["base_url"] = api_config['endpoint']

        client = OpenAI(**client_kwargs)

        model = api_config.get('model', 'gpt-4-turbo-preview')
        max_tokens = api_config.get('max_tokens', self.max_response_tokens)

        if 'gpt-3.5' in model:
            max_tokens = min(max_tokens, 4096)
        elif 'gpt-4' in model:
            max_tokens = min(max_tokens, 8192)  # Most GPT-4 models support up to 8K

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": api_config.get('system_prompt',
                        'You are a helpful assistant that summarizes web content concisely and accurately.')},
                    {"role": "user", "content": prompt}
                ],
                temperature=api_config.get('temperature', 0.7),
                max_tokens=max_tokens,
                timeout=api_config.get('timeout', 60)  # Increased timeout for large content
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")

            result = response.choices[0].message.content
            ai_metadata = {
                'provider': 'openai',
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'finish_reason': response.choices[0].finish_reason,
                'max_tokens_used': max_tokens
            }

            return result, ai_metadata

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

    def _call_openrouter(self, prompt: str, api_config: Dict) -> Tuple[str, Dict]:
        """Call OpenRouter API"""
        api_key = api_config.get('api_key')
        if not api_key:
            raise ValueError("OpenRouter API key not configured")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': api_config.get('referer', 'https://github.com'),
            'X-Title': api_config.get('title', 'Resumeme Script')
        }

        model = api_config.get('model', 'openai/gpt-4-turbo-preview')
        max_tokens = api_config.get('max_tokens', self.max_response_tokens)

        data = {
            'model': model,
            'messages': [
                {"role": "system", "content": api_config.get('system_prompt',
                    'You are a helpful assistant that summarizes web content concisely and accurately.')},
                {"role": "user", "content": prompt}
            ],
            'temperature': api_config.get('temperature', 0.7),
            'max_tokens': max_tokens
        }

        endpoint = api_config.get('endpoint', 'https://openrouter.ai/api/v1/chat/completions')

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=api_config.get('timeout', 60)
            )

            if response.status_code != 200:
                raise ValueError(f"OpenRouter error {response.status_code}: {response.text[:200]}")

            result_json = response.json()
            result = result_json['choices'][0]['message']['content']
            ai_metadata = {
                'provider': 'openrouter',
                'model': result_json.get('model', api_config.get('model')),
                'usage': result_json.get('usage', {}),
                'finish_reason': result_json['choices'][0].get('finish_reason'),
                'max_tokens_used': max_tokens
            }

            return result, ai_metadata

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"OpenRouter API error: {str(e)}") from e

    def _call_ollama(self, prompt: str, api_config: Dict) -> Tuple[str, Dict]:
        """Call local Ollama API"""
        endpoint = api_config.get('endpoint', 'http://localhost:11434/api/chat')
        model = api_config.get('model', 'llama2')

        max_tokens = api_config.get('max_tokens', self.max_response_tokens)

        data = {
            'model': model,
            'messages': [
                {"role": "system", "content": api_config.get('system_prompt',
                    'You are a helpful assistant that summarizes web content concisely and accurately.')},
                {"role": "user", "content": prompt}
            ],
            'stream': False,
            'options': {
                'temperature': api_config.get('temperature', 0.7),
                'num_predict': max_tokens
            }
        }

        try:
            response = requests.post(
                endpoint,
                json=data,
                timeout=api_config.get('timeout', 120)
            )

            if response.status_code != 200:
                raise ValueError(f"Ollama error {response.status_code}: {response.text[:200]}")

            result_json = response.json()
            result = result_json['message']['content']
            ai_metadata = {
                'provider': 'ollama',
                'model': model,
                'done': result_json.get('done', True),
                'max_tokens_used': max_tokens
            }

            return result, ai_metadata

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Ollama API error: {str(e)}") from e

    def _call_anthropic(self, prompt: str, api_config: Dict) -> Tuple[str, Dict]:
        """Call Anthropic Claude API"""
        api_key = api_config.get('api_key')
        if not api_key:
            raise ValueError("Anthropic API key not configured")

        try:
            client = anthropic.Anthropic(api_key=api_key)

            model = api_config.get('model', 'claude-3-opus-20240229')
            max_tokens = api_config.get('max_tokens', self.max_response_tokens)

            if 'opus' in model:
                max_tokens = min(max_tokens, 4096)
            elif 'sonnet' in model or 'haiku' in model:
                max_tokens = min(max_tokens, 4096)

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=api_config.get('temperature', 0.7),
                system=api_config.get('system_prompt',
                    'You are a helpful assistant that summarizes web content concisely and accurately.'),
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text
            ai_metadata = {
                'provider': 'anthropic',
                'model': response.model,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                },
                'max_tokens_used': max_tokens
            }

            return result, ai_metadata

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Anthropic API error: {str(e)}") from e

    def _call_generic_api(self, prompt: str, api_config: Dict) -> Tuple[str, Dict]:
        """Call generic API compatible with OpenAI format"""
        endpoint = api_config.get('endpoint')
        if not endpoint:
            raise ValueError("Endpoint not configured for generic API")

        headers = {
            'Content-Type': 'application/json',
            **api_config.get('headers', {})
        }

        api_key = api_config.get('api_key')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        model = api_config.get('model', 'gpt-3.5-turbo')
        max_tokens = api_config.get('max_tokens', self.max_response_tokens)

        data = {
            'model': model,
            'messages': [
                {"role": "system", "content": api_config.get('system_prompt',
                    'You are a helpful assistant that summarizes web content concisely and accurately.')},
                {"role": "user", "content": prompt}
            ],
            'temperature': api_config.get('temperature', 0.7),
            'max_tokens': max_tokens
        }

        data.update(api_config.get('extra_params', {}))

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=api_config.get('timeout', 60)
            )

            if response.status_code != 200:
                raise ValueError(f"API error {response.status_code}: {response.text[:200]}")

            try:
                result_json = response.json()

                if 'choices' in result_json and len(result_json['choices']) > 0:
                    if 'message' in result_json['choices'][0]:
                        result = result_json['choices'][0]['message']['content']
                    elif 'text' in result_json['choices'][0]:
                        result = result_json['choices'][0]['text']
                    else:
                        result = str(result_json['choices'][0])
                elif 'content' in result_json:
                    result = result_json['content']
                elif 'response' in result_json:
                    result = result_json['response']
                else:
                    result = str(result_json)
            except (KeyError, IndexError, ValueError):
                result = response.text

            ai_metadata = {
                'provider': api_config.get('provider', 'generic'),
                'model': api_config.get('model', 'unknown'),
                'raw_response': response.text[:500],
                'max_tokens_used': max_tokens
            }

            return result, ai_metadata

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Generic API error: {str(e)}") from e

class OutputGenerator:
    """Handles output generation in various formats"""

    def __init__(self, config: Dict[str, Any], quiet_mode: bool = False):
        self.config = config.get('output', {})
        self.quiet_mode = quiet_mode

    def generate_output(self,
                       raw_content: str,
                       cleaned_content: str,
                       ai_result: str,
                       metadata: Dict) -> Dict[str, Any]:
        """Generate output based on configuration"""

        variables = {
            '$IAresult': ai_result,
            '$source_url': metadata.get('url', ''),
            '$timestamp': datetime.now().isoformat(),
            '$word_count': len(cleaned_content.split()),
            '$summary': ai_result[:500] + '...' if len(ai_result) > 500 else ai_result,
            '$title': metadata.get('title', ''),
            '$raw_content': raw_content,
            '$clean_content': cleaned_content,
            '$used_provider': metadata.get('used_provider', ''),
            '$model': metadata.get('ai_metadata', {}).get('model', ''),
            '$total_raw_chars': len(raw_content),
            '$total_clean_chars': len(cleaned_content),
            '$compression_ratio': metadata.get('compression_ratio', 1.0),
            '$prompt_tokens': metadata.get('ai_metadata', {}).get('usage', {}).get('prompt_tokens', 0),
            '$completion_tokens': metadata.get('ai_metadata', {}).get('usage', {}).get('completion_tokens', 0),
            '$total_tokens': metadata.get('ai_metadata', {}).get('usage', {}).get('total_tokens', 0)
        }

        output_fields = self.config.get('fields', {})

        output = {}
        for field_name, field_template in output_fields.items():
            if field_template in variables:
                output[field_name] = variables[field_template]
            elif field_template.startswith('$'):
                var_name = field_template
                output[field_name] = variables.get(var_name, field_template)
            else:
                output[field_name] = field_template

        output['_metadata'] = {
            'processing_time': datetime.now().isoformat(),
            'urls_processed': metadata.get('total_sources', 0),
            'total_raw_characters': len(raw_content),
            'total_clean_characters': len(cleaned_content),
            'compression_ratio': metadata.get('compression_ratio', 1.0),
            'ai_provider': metadata.get('used_provider', 'unknown'),
            'ai_model': metadata.get('ai_metadata', {}).get('model', 'unknown'),
            'ai_tokens_used': {
                'prompt': metadata.get('ai_metadata', {}).get('usage', {}).get('prompt_tokens', 0),
                'completion': metadata.get('ai_metadata', {}).get('usage', {}).get('completion_tokens', 0),
                'total': metadata.get('ai_metadata', {}).get('usage', {}).get('total_tokens', 0)
            },
            'success': bool(ai_result)
        }

        return output

    def save_output(self, output_data: Dict[str, Any]):
        """Save output to file based on configuration, or print to console if no file specified"""
        output_type = self.config.get('type', 'json').lower()
        output_file = self.config.get('file', '')
        encoding = self.config.get('encoding', 'utf-8')

        if not output_file or output_file.strip() == '':
            if output_type == 'json':
                indent = self.config.get('json_indent', 2)
                output_str = json.dumps(output_data, indent=indent, ensure_ascii=False, default=str)
                print(output_str)
                return "console"
            elif output_type in ('text', 'txt'):
                lines = []
                for key, value in output_data.items():
                    if not key.startswith('_'):
                        lines.append(f"{key}: {value}")
                output_str = '\n'.join(lines)
                print(output_str)
                return "console"
            else:
                raise ValueError(f"Unsupported output type: {output_type}")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_type == 'json':
            indent = self.config.get('json_indent', 2)
            with open(output_file, 'w', encoding=encoding) as f:
                json.dump(output_data, f, indent=indent, ensure_ascii=False, default=str)

        elif output_type in ('text', 'txt'):
            with open(output_file, 'w', encoding=encoding) as f:
                lines = []
                for key, value in output_data.items():
                    if not key.startswith('_'):
                        lines.append(f"{key}: {value}")
                f.write('\n'.join(lines))
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        if not self.quiet_mode:
            logging.info("Output saved to %s", output_file)
        return output_file

class Resumeme:
    """Main class orchestrating the entire process"""

    def __init__(self, config_path: str, prompt_override: str = None, quiet_mode: bool = False):
        self.config_path = config_path
        self.config = None
        self.prompt_override = prompt_override
        self.quiet_mode = quiet_mode

    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})

        if log_config.get('enabled', True):
            log_level = getattr(logging, log_config.get('level', 'INFO').upper())
            log_file = log_config.get('file', 'resumeme.log')

            handlers = [logging.FileHandler(log_file, encoding='utf-8')]
            if not self.quiet_mode:
                handlers.append(logging.StreamHandler())

            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=handlers
            )

            if not self.quiet_mode:
                logging.info("Logging initialized. Configuration: %s", self.config_path)

    def run(self):
        """Main execution method"""

        config_loader = ConfigLoader(self.config_path)
        self.config = config_loader.load()

        if self.prompt_override:
            if self.prompt_override in self.config.get('prompt_templates', {}):
                self.config['selected_prompt'] = self.prompt_override
                if not self.quiet_mode:
                    logging.info("Using command line prompt: %s", self.prompt_override)
            else:
                available_prompts = list(self.config.get('prompt_templates', {}).keys())
                logging.warning("Prompt '%s' not found in config. Available prompts: %s",
                              self.prompt_override, available_prompts)
                if not self.quiet_mode:
                    print(f"Warning: Prompt '{self.prompt_override}' not found. Using default.")

        self.setup_logging()

        if not self.quiet_mode:
            logging.info("Starting resumeme process")

        scraper = WebScraper(self.config)
        content_list = scraper.fetch_all_urls()

        successful_content = [(content, metadata) for content, metadata in content_list
                            if content is not None]

        if not successful_content:
            print("Error: Failed to fetch content from all URLs")
            sys.exit(1)

        total_raw_size = sum(len(content) for content, _ in successful_content)
        logging.info("Total raw content: %d chars (~%.1f KB)",
                    total_raw_size, total_raw_size / 1024)

        cleaner = HTMLCleaner(self.config)
        cleaned_content_list = []
        raw_content_list = []

        for content, metadata in successful_content:
            cleaned_content, updated_metadata = cleaner.clean_html(content, metadata)
            cleaned_content_list.append((cleaned_content, updated_metadata))
            raw_content_list.append((content, metadata))

        total_clean_size = sum(len(content) for content, _ in cleaned_content_list)
        avg_compression = total_raw_size / total_clean_size if total_clean_size > 0 else 1
        logging.info("Total clean content: %d chars (~%.1f KB, compression: %.2fx)",
                    total_clean_size, total_clean_size / 1024, avg_compression)

        processing_config = self.config.get('processing', {})
        concat_strategy = processing_config.get('concat_strategy', 'sequential')

        if concat_strategy == 'sequential':
            all_raw_content = '\n\n'.join([content for content, _ in raw_content_list])
            all_clean_content = '\n\n'.join([content for content, _ in cleaned_content_list])
        elif concat_strategy == 'chunked':
            chunk_size = processing_config.get('chunk_size', 30000)
            all_raw_content = self._chunk_content(raw_content_list, chunk_size)
            all_clean_content = self._chunk_content(cleaned_content_list, chunk_size)
        else:
            all_raw_content = '\n\n'.join([content for content, _ in raw_content_list])
            all_clean_content = '\n\n'.join([content for content, _ in cleaned_content_list])

        max_length = processing_config.get('max_total_length', 150000)  # 150K chars ~ 37.5K tokens
        if len(all_clean_content) > max_length:
            logging.warning("Content too long (%d chars), truncating to %d chars",
                          len(all_clean_content), max_length)
            all_clean_content = all_clean_content[:max_length] + "\n\n[Content truncated due to length limit]"

        logging.info("Final content for AI: %d chars (~%d tokens)",
                    len(all_clean_content), len(all_clean_content) // 4)

        ai_processor = AIProcessor(self.config)
        ai_result, final_metadata = ai_processor.process_content(all_clean_content,
            {'total_sources': len(cleaned_content_list)})

        output_generator = OutputGenerator(self.config, self.quiet_mode)

        combined_metadata = {
            'url': '; '.join([m.get('url', '') for _, m in cleaned_content_list]),
            'sources': [metadata for _, metadata in cleaned_content_list],
            'ai_processing': final_metadata.get('ai_metadata', {}),
            'used_provider': final_metadata.get('used_provider', 'none'),
            'total_sources': len(cleaned_content_list),
            'total_raw_characters': len(all_raw_content),
            'total_clean_characters': len(all_clean_content),
            'compression_ratio': avg_compression
        }

        output_data = output_generator.generate_output(
            all_raw_content,
            all_clean_content,
            ai_result,
            combined_metadata
        )

        output_file = output_generator.save_output(output_data)

        if not self.quiet_mode and output_file != "console":
            self._print_summary(cleaned_content_list, ai_result, output_file,
                               final_metadata.get('used_provider', 'unknown'))

        if not self.quiet_mode:
            logging.info("Resumeme process completed successfully")

        return output_data, output_file

    def _chunk_content(self, content_list: List[Tuple[str, Dict]], chunk_size: int) -> str:
        """Split content into chunks for processing"""
        chunks = []
        current_chunk = ""

        for content, _ in content_list:
            if len(current_chunk) + len(content) <= chunk_size:
                current_chunk += "\n\n" + content if current_chunk else content
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = content

        if current_chunk:
            chunks.append(current_chunk)

        return "\n\n[CHUNK SEPARATOR]\n\n".join(chunks)

    def _print_summary(self, content_list: List[Tuple[str, Dict]], ai_result: str,
                      output_file: str, used_provider: str):
        """Print a summary of the process to console"""

        if self.quiet_mode or output_file == "console":
            return

        print("\n" + "="*60)
        print("RESUMEME PROCESS COMPLETED")
        print("="*60)

        total_raw = sum(metadata.get('size_bytes', 0) for _, metadata in content_list)
        total_clean = sum(metadata.get('cleaned_length', 0) for _, metadata in content_list)
        compression = total_raw / total_clean if total_clean > 0 else 1

        print(f"\nSources processed: {len(content_list)}")
        print(f"Total raw size: {total_raw:,} bytes ({total_raw/1024:.1f} KB)")
        print(f"Total clean size: {total_clean:,} chars ({total_clean/1024:.1f} KB)")
        print(f"Compression ratio: {compression:.2f}x")

        for i, (_content, metadata) in enumerate(content_list, 1):
            status = "✓" if metadata.get('status_code') == 200 else "✗"
            size_kb = metadata.get('size_bytes', 0) / 1024
            clean_kb = metadata.get('cleaned_length', 0) / 1024
            print(f"  {status} [{i}] {metadata.get('url', 'Unknown')}")
            print(f"     Size: {size_kb:.1f}KB → {clean_kb:.1f}KB ({metadata.get('compression_ratio', 1):.2f}x)")
            if 'error' in metadata:
                print(f"     Error: {metadata['error'][:100]}...")

        print(f"\nAI provider used: {used_provider}")
        print("\nAI Result (preview):")
        print("-"*40)
        preview = ai_result[:500] + "..." if len(ai_result) > 500 else ai_result
        print(preview)
        print("-"*40)

        print(f"\nOutput saved to: {output_file}")
        print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Web content scraper and AI summarizer with multi-API fallback"
    )
    parser.add_argument(
        "config",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional prompt template name to use (overrides config)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: suppress all console output when output file is specified"
    )

    args = parser.parse_args()

    config_file = args.config
    prompt_override = args.prompt
    quiet_mode = args.quiet

    try:
        resumeme = Resumeme(config_file, prompt_override, quiet_mode)
        resumeme.run()
    except KeyboardInterrupt:
        if not quiet_mode:
            print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error: {e}")
        logging.error("Unhandled exception: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
