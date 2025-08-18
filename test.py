# test.py
# import os, asyncio, json
# from mcp_connection.manager import MCPManager, MCPServer

# os.environ["MCP_SERVERS"] = '''
# [
#   {"name":"yfinance","command":"/workspaces/new_test/.venv/bin/python -m yahoo_finance_mcp.server"},
#   {"name":"financial-datasets","command":"/workspaces/new_test/.venv/bin/python /workspaces/new_test/vendors/financial-datasets-mcp/server.py",
#    "env":{"FINANCIAL_DATASETS_API_KEY":"9853d5a8-5dfd-4034-b2c2-653bc6868d7f"}}
# ]'''

# async def main():
#     mgr = MCPManager.instance()
#     await mgr.start_all(MCPServer.from_env())
#     print("tools@yfinance:", await mgr.list_tools("yfinance"))
#     print("tools@financial-datasets:", await mgr.list_tools("financial-datasets"))

# asyncio.run(main())


# /workspaces/new_test/test.py
import os, sys, asyncio, json
from mcp_connection.manager import MCPManager, MCPServer

os.environ["MCP_SERVERS"] = json.dumps([
  {"name": "yfinance", "command": "python vendors/yahoo-finance-mcp/server.py"},
  {"name": "financial-datasets", "command": "python vendors/financial-datasets-mcp/server.py",
   "env": {"FINANCIAL_DATASETS_API_KEY": "9853d5a8-5dfd-4034-b2c2-653bc6868d7f"}}
])

async def main():
    mgr = MCPManager.instance()
    servers = MCPServer.from_env()
    await mgr.start_all(servers)
    for name in servers.keys():
        print(name, await mgr.list_tools(name))

asyncio.run(main())
