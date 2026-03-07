#!/usr/bin/env bash
# ============================================================
# OwnAI – One-click installer for Linux / macOS
# Usage: bash setup.sh
# ============================================================
set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${BOLD}🧠  OwnAI Setup${NC}"
echo "────────────────────────────────────"

# ── Check Python ────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+ first.${NC}"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python ${PY_VER} found at ${PYTHON}"

# ── Create virtual environment ──────────────────────────────
if [ ! -d "venv" ]; then
    echo -e "\n${BOLD}Creating virtual environment…${NC}"
    $PYTHON -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment ready"

# ── Upgrade pip ─────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Install dependencies ────────────────────────────────────
echo -e "\n${BOLD}Installing dependencies (this may take a few minutes)…${NC}"
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓${NC} Dependencies installed"

# ── Copy .env if missing ────────────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env from .env.example"
    echo -e "  ${YELLOW}→ Edit .env to set your SECRET_KEY and other settings${NC}"
fi

# ── Check GPU ───────────────────────────────────────────────
echo ""
$PYTHON check_gpu.py

# ── Done ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✅  Setup complete!${NC}"
echo ""
echo "To start OwnAI:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open  http://localhost:5000  in your browser."
echo ""
