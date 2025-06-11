# Batch Verification Processing

This directory contains tools for processing verification tasks using Azure OpenAI's batch API.

## Files

- `batch_processor.py` - Main script for processing multiple JSONL files in batches
- `prepare_batch.ipynb` - Original notebook for single batch processing
- `example_batch_input.jsonl` - Example input format

## Setup

1. Set your Azure OpenAI API key:
```bash
export AZURE_API_KEY="your-api-key-here"
```

2. Create a `verification_batches/` directory with your JSONL files:
```bash
mkdir verification_batches
# Copy your JSONL files to this directory
```

## Usage

### Automated Batch Processing

Process all JSONL files in the `verification_batches/` directory:

```bash
python batch_processor.py
```

### Configuration Options

You can customize the processing by modifying the parameters:

```python
processor = BatchProcessor(
    verification_batches_dir="verification_batches",  # Input directory
    max_concurrent_batches=5,                         # Max simultaneous batches
    max_retries=10,                                   # Retry attempts for token limits
    azure_endpoint="https://your-endpoint.com/",     # Azure OpenAI endpoint
    api_key="your-api-key"                           # API key (or use env var)
)
```

### Multiple Deployments (Parallel Processing)

To use multiple Azure deployments for increased throughput:

```python
# Set up environment variables for different deployments
export AZURE_API_KEY_1="key-for-deployment-1"
export AZURE_API_KEY_2="key-for-deployment-2"

# Create separate directories for each deployment
mkdir verification_batches_o4-mini
mkdir verification_batches_o3-mini

# Run parallel processors
python -c "from batch_processor import run_parallel_processors; run_parallel_processors()"
```

Or create processors manually:

```python
from batch_processor import BatchProcessor

# Processor for o4-mini deployment
processor1 = BatchProcessor(
    verification_batches_dir="verification_batches_o4",
    azure_endpoint="https://aisg-sj.openai.azure.com/",
    api_key="your-o4-key"
)

# Processor for o3-mini deployment  
processor2 = BatchProcessor(
    verification_batches_dir="verification_batches_o3",
    azure_endpoint="https://decla-mbncunfi-australiaeast.cognitiveservices.azure.com/",
    api_key="your-o3-key"
)
```

### Key Features

- **Automatic Queue Management**: Maintains optimal number of concurrent batches
- **Token Limit Handling**: Automatically retries with exponential backoff when hitting TPM limits
- **Progress Monitoring**: Real-time status updates every 30 seconds
- **Result Processing**: Automatically saves results as `verification_results_{filename}.json`
- **Error Handling**: Tracks failed jobs and provides detailed error reporting

### Input Format

Each JSONL file should contain one JSON object per line, following the format in `example_batch_input.jsonl`:

```json
{"custom_id":"request-1","method":"POST","url":"/chat/completions","body":{...}}
```

### Output

Results are saved as `verification_results_{input_filename}.json` with the format:

```json
[
  {
    "custom_id": "request-1",
    "verification_response": "Generated response content..."
  }
]
```

### Monitoring

The script provides real-time updates:
- 📁 File discovery
- 📤 File uploads  
- 🚀 Batch creation
- ⏳ Status monitoring
- 🎉 Completion notifications
- 💥 Error reporting

### Error Recovery

- Automatic retry for token limit exceeded errors
- Exponential backoff to avoid API rate limits
- Detailed logging of failed operations
- Final summary of successful vs failed jobs 