#!/bin/bash
# Fix MCP Setup Script

echo "🔧 Fixing MCP Server Setup..."

# 1. First, let's check what's in your yahoo-finance-mcp directory
echo "Checking yahoo-finance-mcp directory..."
ls -la vendors/yahoo-finance-mcp/

# 2. Install the correct yahoo finance MCP package
echo "Installing correct yahoo-finance MCP package..."
pip install mcp-yahoo-finance

# 3. Alternative: Install from the repository you cloned
echo "Installing from local repository..."
cd vendors/yahoo-finance-mcp
pip install -e .
cd ../..

# 4. Fix the .env file with correct commands
echo "Fixing .env configuration..."

# Create backup of .env
cp .env .env.backup

# Update MCP_SERVERS in .env with correct configuration
cat << 'EOF' > temp_mcp_config.json
[
  {
    "name": "yfinance",
    "command": "/workspaces/new_test/.venv/bin/python -m mcp_yahoo_finance.server"
  },
  {
    "name": "financial-datasets",
    "command": "/workspaces/new_test/.venv/bin/python /workspaces/new_test/vendors/financial-datasets-mcp/server.py",
    "env": {
      "FINANCIAL_DATASETS_API_KEY": "$FINANCIAL_DATASETS_API_KEY"
    }
  }
]
EOF

# Read the JSON and escape it for .env
MCP_SERVERS_JSON=$(cat temp_mcp_config.json | jq -c .)
rm temp_mcp_config.json

# Update .env file
sed -i.bak "s|^MCP_SERVERS=.*|MCP_SERVERS='${MCP_SERVERS_JSON}'|" .env

echo "✅ MCP Setup Fixed!"
echo ""
echo "Updated .env with correct MCP server configurations:"
echo "- yfinance: mcp_yahoo_finance.server module"
echo "- financial-datasets: local server script"
echo ""
echo "To test, run: python setup_and_run.py"