# run.sh - start CLI or Streamlit web app
MODE=${1:-web} # web or cli
PROD_ID=${2:-101}
TOP_N=${3:-5}

# Activate venv if present (POSIX)
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  . .venv/bin/activate
fi

echo "Mode: $MODE  product: $PROD_ID  top: $TOP_N"

if [ "$MODE" = "cli" ]; then
  python main.py --product "$PROD_ID" --top "$TOP_N"
else
  # launch Streamlit web app
  streamlit run app.py -- --product "$PROD_ID" --top "$TOP_N"
fi
