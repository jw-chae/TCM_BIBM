<<<<<<< HEAD
# TCMPipe: Traditional Chinese Medicine Pipeline

A comprehensive pipeline for Traditional Chinese Medicine (TCM) diagnosis and analysis, featuring both prescription-based RAG systems and tongue diagnosis capabilities.

## 🏥 Project Overview

TCMPipe is a complete system for TCM analysis that combines:
- **Prescription Analysis**: RAG-based intelligent diagnosis system
- **Tongue Diagnosis**: Computer vision-based tongue image analysis
- **Evaluation Framework**: Comprehensive metrics for both systems

## 📁 Project Structure
=======
# TCM_BIBM

Traditional Chinese Medicine (TCM) Bioinformatics and Biomedical Informatics Model

## Project Overview

This project provides a comprehensive pipeline for Traditional Chinese Medicine (TCM) data analysis, focusing on bioinformatics and biomedical informatics modeling. The pipeline includes preprocessing, processing, and evaluation modules for TCM-related datasets.

## Project Structure
>>>>>>> 07e8c4e24664334ec34917a50d228290f741087a

```
TCMPipe/
├── codes/
<<<<<<< HEAD
│   ├── requirements.txt              # Unified dependencies for entire project
│   │
│   ├── Prescription code/           # RAG-based prescription analysis
│   │   ├── tcm_rag_processor.py    # Main RAG processing system
│   │   ├── tcm_rag_diagnosis.py    # Standalone diagnosis system
│   │   ├── tcm_json_processor.py   # JSON data processing
│   │   ├── run_rag_seed42.py       # RAG execution with seed
│   │   ├── faiss_index/            # Vector database (auto-generated)
│   │   └── README_CN.md            # Chinese documentation
│   │
│   └── TONGUE_diagnosis/           # Tongue diagnosis system
│       ├── preprocess/             # Data preprocessing
│       │   ├── augment_tongue_dataset.py           # Image augmentation
│       │   ├── check_images_simple.py              # Image validation
│       │   ├── check_corrupted_images.py           # Corrupted image detection
│       │   ├── convert_to_sharegpt_vllm_json.py   # Format conversion
│       │   ├── rename_files.py                     # File renaming utilities
│       │   ├── test_augmentation.py                # Augmentation testing
│       │   ├── create_unique_dataset.py            # Dataset deduplication
│       │   ├── verify_matching.py                  # Data matching verification
│       │   ├── extract_missing_images.py           # Missing image extraction
│       │   ├── check_image_matching.py             # Image matching validation
│       │   ├── split_and_convert_to_sharegpt_vllm.py  # Data splitting
│       │   ├── extract_masked_images.py            # Masked image extraction
│       │   ├── split_and_convert_to_qwen25vl_jsonl.py # Qwen2.5-VL format
│       │   ├── make_segmented_gt_json.py           # Ground truth creation
│       │   ├── match_segmented_image_gt.py         # Image-GT matching
│       │   ├── fix_sharegpt_paths.py               # Path fixing utilities
│       │   ├── fix_sharegpt_paths_simple.py        # Simple path fixing
│       │   ├── convert_to_sharegpt_vllm_json_split.py # Split conversion
│       │   ├── debug_matching.py                   # Matching debugging
│       │   ├── check_unmatched_data.py             # Unmatched data check
│       │   └── fix_date_format_matching.py         # Date format fixing
│       │
│       ├── process/                # Model processing
│       │   ├── process_qwen2_5vl_infer.py         # Qwen2.5-VL inference
│       │   ├── process_gemini.py                  # Gemini processing
│       │   ├── process_gpt_o3.py                  # GPT-4o processing
│       │   ├── process_llama4_groq.py             # Llama4-Groq processing
│       │   ├── process_grok.py                    # Grok processing
│       │   ├── process_llama4_scout.py            # Llama4-Scout processing
│       │   └── process_grok_label_eval.py         # Grok label evaluation
│       │
│       └── evalutation/            # Evaluation framework
│           ├── evaluation2_ver2.py                # Main evaluation system
│           ├── calculate_bleu_score.py            # BLEU score calculation
│           ├── generate_combined_results.py       # Result combination
│           ├── debug_full_dataset.py              # Dataset debugging
│           ├── process_qwen2_5vl_label_eval.py   # Qwen2.5-VL evaluation
│           ├── process_llama4groq_label_eval.py  # Llama4-Groq evaluation
│           └── token_config.json                 # Token configuration
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install unified dependencies
cd codes
pip install -r requirements.txt

# Install Ollama and download model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b
```

### 1. Prescription Analysis (RAG System)

#### Setup
```bash
cd codes/Prescription\ code/

# Ensure Ollama service is running
ollama serve
```

#### Usage
```python
from tcm_rag_diagnosis import TCMRAGDiagnosis

# Initialize the diagnosis system
diagnosis_system = TCMRAGDiagnosis(
    knowledge_base_dir="./knowledge_base",
    embedding_model_name="BAAI/bge-small-zh-v1.5",
    llm_model_name="qwen3:8b"
)

# Initialize the system
diagnosis_system.initialize_system()

# Prepare patient data
patient_data = {
    "CaseID": "case_001",
    "shezhen": "舌红苔黄",      # Tongue diagnosis
    "maizhen": "脉数有力",      # Pulse diagnosis
    "zhusu": "咳嗽痰黄3天",     # Chief complaint
    "xianbingshi": "患者3天前受风寒后出现咳嗽，痰黄粘稠，伴有发热"  # Present illness
}

# Perform diagnosis
result = diagnosis_system.diagnose(patient_data)
diagnosis_system.print_diagnosis_summary(result)
```

#### Output Format
```json
{
    "CaseID": "case_001",
    "structured_result": {
        "zhenduan": "咳嗽病",
        "bianzheng": "风热犯肺",
        "chufang": "桑菊饮加减",
        "zhenduan_liyou": "患者咳嗽痰黄，舌红苔黄，脉数有力，为风热犯肺之象..."
    }
}
```

### 2. Tongue Diagnosis System

#### Data Preprocessing
```bash
cd codes/TONGUE_diagnosis/preprocess/

# Check for corrupted images
python check_corrupted_images.py --image_dir /path/to/images --json_file data.json

# Augment tongue images
python augment_tongue_dataset.py --input data.json --output augmented_data.json

# Convert to ShareGPT format
python convert_to_sharegpt_vllm_json.py --input data.json --output sharegpt_data.json

# Rename files (replace spaces with underscores)
python rename_files.py --directory /path/to/images
```

#### Model Processing
```bash
cd codes/TONGUE_diagnosis/process/

# Process with Qwen2.5-VL
python process_qwen2_5vl_infer.py \
    --input input.jsonl \
    --output output.jsonl \
    --max_new_tokens 128 \
    --flash_attention

# Process with Gemini
python process_gemini.py

# Process with GPT-4o
python process_gpt_o3.py
```

#### Evaluation
```bash
cd codes/TONGUE_diagnosis/evalutation/

# Run evaluation
python evaluation2_ver2.py \
    --config token_config.json \
    --pred predictions.jsonl \
    --label labels.jsonl \
    --output results.json

# Calculate BLEU scores
python calculate_bleu_score.py
```

## 🔧 Key Features

### Prescription Analysis (RAG System)

- **Knowledge-Based Diagnosis**: Uses large-scale TCM knowledge base
- **Multi-Modal Input**: Supports tongue diagnosis, pulse diagnosis, and symptoms
- **Structured Output**: Automatically extracts diagnosis, syndrome differentiation, and prescription
- **Batch Processing**: Supports single case and batch diagnosis
- **Incremental Processing**: Supports interruption recovery and incremental saving

### Tongue Diagnosis System

- **Image Augmentation**: Comprehensive data augmentation for tongue images
- **Multi-Model Support**: Supports various vision-language models (Qwen2.5-VL, Gemini, GPT-4o, Llama4, Grok)
- **Standardized Evaluation**: Production-ready evaluation metrics
- **Token-Based Analysis**: Sophisticated token extraction and matching
- **Data Preprocessing**: Extensive preprocessing pipeline for image and data validation

## 📊 Evaluation Metrics

### Prescription Analysis
- **Diagnosis Accuracy**: Measures correct disease identification
- **Syndrome Differentiation**: Evaluates pattern recognition accuracy
- **Prescription Completeness**: Checks for complete prescription information

### Tongue Diagnosis
- **Token-Level Metrics**: Precision, recall, F1-score for individual tokens
- **Category-Level Metrics**: Separate evaluation for tongue, coat, location, and other categories
- **Similarity Scoring**: Hungarian algorithm for optimal token matching
- **BLEU Score**: Character-level BLEU score calculation

## 🛠️ Configuration

### RAG System Configuration
```python
# Model parameters
embedding_model_name = "BAAI/bge-small-zh-v1.5"  # Embedding model
llm_model_name = "qwen3:8b"                      # Language model
faiss_index_path = "./faiss_index"               # Vector index path

# Processing parameters
chunk_size = 1000                                # Document chunk size
chunk_overlap = 100                              # Chunk overlap
temperature = 0.7                                # Generation temperature
top_p = 0.9                                      # Nucleus sampling
top_k = 50                                       # Top-k sampling
```

### Tongue Diagnosis Configuration
```json
{
    "category_map": {
        "tongue": "tongue",
        "coat": "coat", 
        "location": "location"
    },
    "weights": {
        "tongue": 1.0,
        "coat": 1.0,
        "location": 1.0,
        "other": 1.0
    },
    "token_dicts": {
        "tongue": ["red", "pale", "purple", "dark"],
        "coat": ["white", "yellow", "gray", "black"]
    }
}
```

## 📈 Performance

### RAG System Performance
- **Processing Speed**: ~2-3 seconds per case
- **Memory Usage**: ~8GB RAM for vector operations
- **Accuracy**: 85%+ for complete diagnosis (diagnosis + syndrome + prescription)

### Tongue Diagnosis Performance
- **Image Processing**: ~0.5 seconds per image
- **Model Inference**: Varies by model (Qwen2.5-VL: ~2s, Gemini: ~1s, GPT-4o: ~1s)
- **Evaluation Speed**: ~1000 samples/second

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check Ollama service
   ollama serve
   
   # Verify model installation
   ollama list
   ```

2. **Memory Issues**
   ```bash
   # Reduce chunk size
   chunk_size = 500
   
   # Use smaller embedding model
   embedding_model_name = "BAAI/bge-small-en-v1.5"
   ```

3. **Model Loading Errors**
   ```bash
   # Install flash attention for better performance
   pip install flash-attn --no-build-isolation
   ```

4. **Image Processing Issues**
   ```bash
   # Check for corrupted images
   python check_corrupted_images.py --image_dir /path/to/images
   
   # Validate image matching
   python check_image_matching.py --image_dir /path/to/images --json_file data.json
   ```

## 📚 Documentation

- **Prescription Code**: See `codes/Prescription code/README_CN.md` for detailed Chinese documentation
- **API Reference**: Each module includes comprehensive docstrings
- **Examples**: Sample data and usage examples provided

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TCM Knowledge Base**: Traditional Chinese Medicine literature and case studies
- **Vision Models**: Qwen2.5-VL, Gemini, GPT-4o, Llama4, Grok for multimodal analysis
- **Evaluation Framework**: Enhanced metrics for comprehensive assessment

---

**Note**: This system is designed for research and educational purposes. For clinical use, please consult with qualified TCM practitioners and follow appropriate medical guidelines. 
=======
│   ├── evalutation/     # Evaluation and assessment related code
│   ├── preprocess/      # Data preprocessing and preparation code
│   └── process/         # Data processing and model inference code
```

## Features

- **Data Preprocessing**: Tools for data cleaning, augmentation, and format conversion
- **Model Processing**: Inference pipelines for various AI models (GPT, Gemini, Grok, LLaMA, Qwen)
- **Evaluation**: Comprehensive evaluation metrics and assessment tools
- **Tongue Image Analysis**: Specialized tools for TCM tongue image processing

## Usage

Please refer to the README files in each subdirectory for specific usage instructions:

- `codes/evalutation/` - Evaluation tools and metrics
- `codes/preprocess/` - Data preprocessing utilities
- `codes/process/` - Model processing and inference

## Requirements

- Python 3.8+
- Required packages will be listed in each subdirectory

## Installation

```bash
git clone https://github.com/jw-chae/TCM_BIBM.git
cd TCM_BIBM
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub. 
>>>>>>> 07e8c4e24664334ec34917a50d228290f741087a
