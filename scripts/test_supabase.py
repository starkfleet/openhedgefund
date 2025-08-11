#!/usr/bin/env python3
"""Test Supabase connection and schema."""

import asyncio
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aggregator.supabase_writer import get_db_pool


async def test_connection():
    """Test database connection and basic operations."""
    try:
        print("Connecting to Supabase...")
        pool = await get_db_pool()
        
        async with pool.acquire() as conn:
            # Test basic connection
            version = await conn.fetchval("SELECT version()")
            print(f"✓ Connected to: {version}")
            
            # Test schema and tables
            tables = await conn.fetch("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'market_data'
                ORDER BY table_name
            """)
            
            if tables:
                print("✓ Found tables in market_data schema:")
                for table in tables:
                    print(f"  - {table['table_name']} ({table['table_type']})")
            else:
                print("✗ No tables found in market_data schema")
                return False
            
            print("\n✓ All tests passed! Supabase (Postgres) is ready.")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        if 'pool' in locals():
            await pool.close()


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
