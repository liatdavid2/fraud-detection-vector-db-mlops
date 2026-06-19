from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SAMPLE_FEATURES = {
    "income": 18000,
    "name_email_similarity": 0.18,
    "prev_address_months_count": -1,
    "current_address_months_count": 12,
    "customer_age": 22,
    "days_since_request": 0.01,
    "intended_balcon_amount": -1,
    "payment_type": "AE",
    "zip_count_4w": 1800,
    "velocity_6h": 72,
    "velocity_24h": 520,
    "velocity_4w": 6200,
    "bank_branch_count_8w": 1,
    "date_of_birth_distinct_emails_4w": 12,
    "employment_status": "CA",
    "credit_risk_score": 80,
    "email_is_free": 1,
    "housing_status": "BB",
    "phone_home_valid": 0,
    "phone_mobile_valid": 1,
    "bank_months_count": 1,
    "has_other_cards": 0,
    "proposed_credit_limit": 2000,
    "foreign_request": 1,
    "source": "INTERNET",
    "session_length_in_minutes": 2.1,
    "device_os": "windows",
    "keep_alive_session": 0,
    "device_distinct_emails_8w": 2,
    "device_fraud_count": 0,
    "month": 7,
}


def print_result(title: str, result) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for item in result.content:
        text = getattr(item, "text", None)
        if text is not None:
            try:
                parsed = json.loads(text)
                print(json.dumps(parsed, indent=2))
            except Exception:
                print(text)
        else:
            print(item)


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_path};{env.get('PYTHONPATH', '')}"

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "fraud_vector_db_mlops.mcp_server"],
        env=env,
        cwd=str(project_root),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("\nAvailable MCP tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            summary = await session.call_tool(
                "get_latest_training_summary",
                arguments={},
            )
            print_result("get_latest_training_summary", summary)

            prediction = await session.call_tool(
                "predict_fraud",
                arguments={"features": SAMPLE_FEATURES},
            )
            print_result("predict_fraud", prediction)

            # This requires Milvus to be running.
            similar = await session.call_tool(
                "find_similar_fraud_cases",
                arguments={"features": SAMPLE_FEATURES, "top_k": 50},
            )
            print_result("find_similar_fraud_cases", similar)


if __name__ == "__main__":
    asyncio.run(main())