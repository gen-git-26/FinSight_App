# setup_and_run.py
"""
FinSight Setup and Run Script
Helps set up and verify the MCP connections before running the main application.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_path = Path(".env")
    if not env_path.exists():
        print(".env file not found!")
        return False
    
    required_vars = [
        "OPENAI_API_KEY",
        "FINNHUB_API_KEY", 
        "ALPHAVANTAGE_API_KEY",
        "FINANCIAL_DATASETS_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "MCP_SERVERS"
    ]
    
    missing_vars = []
    with open(env_path) as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=\n" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing or empty environment variables: {', '.join(missing_vars)}")
        return False
    
    print(".env file looks good")
    return True

def validate_mcp_servers():
    """Validate MCP_SERVERS configuration."""
    mcp_servers = os.getenv("MCP_SERVERS", "")
    if not mcp_servers:
        print("MCP_SERVERS not configured")
        return False
    
    try:
        servers = json.loads(mcp_servers)
        print(f"Found {len(servers)} MCP servers configured:")
        
        for server in servers:
            name = server.get("name", "unknown")
            command = server.get("command", "")
            print(f"  • {name}: {command}")
            
            # Check if command path exists for Python scripts
            if command.startswith("/workspaces/new_test/.venv/bin/python"):
                python_path = command.split()[0]
                if not Path(python_path).exists():
                    print(f"    Python path doesn't exist: {python_path}")
                else:
                    print(f"    Python path exists")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f" Invalid MCP_SERVERS JSON: {e}")
        return False

def test_mcp_connection():
    """Test MCP server connections."""
    print("\n Testing MCP connections...")
    
    try:
        # Import our modules
        sys.path.append(str(Path.cwd()))
        from mcp_connection.manager import MCPServer
        from mcp_connection.startup import startup_mcp_servers, get_manager
        
        # Start servers
        print("Starting MCP servers...")
        results = startup_mcp_servers()
        
        if not results:
            print("No servers started")
            return False
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"Server startup results: {success_count}/{total_count} successful")
        
        for name, success in results.items():
            status = "All good!" if success else "Somthing went wrong"
            print(f"  {status} {name}")
        
        if success_count > 0:
            print("\n Testing a sample MCP call...")
            try:
                from tools.mcp_router import route_and_call
                result = route_and_call("current price of AAPL")
                if result and not result.startswith("MCP Error:"):
                    print("MCP routing test successful!")
                    print(f"Sample result: {result[:100]}...")
                else:
                    print(f" MCP test returned: {result}")
            except Exception as e:
                print(f" MCP routing test failed: {e}")
        
        # Clean up
        get_manager().stop_all_servers()
        return success_count > 0
        
    except Exception as e:
        print(f" MCP connection test failed: {e}")
        return False

def run_application():
    """Run the Streamlit application."""
    print("\n Starting FinSight application...")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n Application stopped by user")
    except Exception as e:
        print(f" Failed to start application: {e}")

def main():
    """Main setup and run function."""
    print("🔍 FinSight Setup & Validation")
    print("=" * 40)
    
    # Step 1: Check environment
    if not check_env_file():
        print("\n Environment setup incomplete. Please check your .env file.")
        return
    
    # Step 2: Validate MCP configuration  
    if not validate_mcp_servers():
        print("\n MCP server configuration invalid. Please check MCP_SERVERS in .env.")
        return
    
    # Step 3: Test MCP connections
    if not test_mcp_connection():
        print("\n MCP connection test failed. Application will run with limited functionality.")
        input("Press Enter to continue anyway, or Ctrl+C to abort...")
    else:
        print("\n All systems ready!")
    
    # Step 4: Run application
    print("\n" + "=" * 40)
    run_application()

if __name__ == "__main__":
    main()