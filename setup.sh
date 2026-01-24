#!/usr/bin/env bash

# Exit on error
set -e

echo "🚀 Starting llamaindex setup..."

# 1. Setup .env
if [ ! -f .env ]; then
    echo "📄 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️ Please edit .env and add your API keys (OpenAI, Anthropic)."
else
    echo "✅ .env file already exists."
fi

# 2. Setup Main Environment
echo "📦 Setting up main Python environment using uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

if [ ! -d .venv ]; then
    uv venv
    source .venv/bin/activate
    echo "🛠️ Installing project in editable mode..."
    uv pip install -e .
else
    echo "✅ Main environment (.venv) already exists."
fi

# 3. Setup MinerU Isolated Environment
echo "⛏️ Setting up MinerU isolated environment..."
if [ ! -d .mineru_env ]; then
    uv venv .mineru_env
    source .mineru_env/bin/activate
    uv pip install -r requirements_mineru.txt
    deactivate
    echo "✅ MinerU environment created."
else
    echo "✅ MinerU environment already exists."
fi

# 4. Setup Data Directory and Download Demo
echo "📁 Setting up data directory..."
mkdir -p data/paul_graham
if [ ! -f data/paul_graham/paul_graham_essay.pdf ]; then
    echo "📥 Downloading demo PDF (Paul Graham Essay)..."
    curl -L "https://drive.google.com/uc?export=download&id=1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-" -o data/paul_graham/paul_graham_essay.pdf
    echo "✅ Downloaded demo PDF to ./data/paul_graham/"
else
    echo "✅ Demo PDF already exists."
fi

echo ""
echo "✨ Setup complete!"
echo "Next steps:"
echo "1. Edit .env and add your API keys."
echo "2. Start all databases:  bash db.sh start_all"
echo "3. Run the RAG system:  python main.py"
