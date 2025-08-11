# Supabase Setup Guide

## 1. Create Supabase Project
1. Go to https://supabase.com
2. Sign up/login and create a new project
3. Choose a name (e.g., "openhedgefund")
4. Set a database password (save this!)
5. Choose a region close to you
6. Wait for project to be ready (2-3 minutes)

## 2. Get Connection Details
1. Go to Project Settings > Database
2. Copy the "Connection string" (URI format)
3. Note: Host, Database name, Port, User, Password

## 3. Update .env
Add to your .env file:
```
SUPABASE_DB_URL=postgresql://postgres:[YOUR-PASSWORD]@[HOST]:5432/postgres
```

## 4. Run SQL Setup
1. Go to SQL Editor in Supabase dashboard
2. Run the SQL from `supabase_migration.sql`
3. Verify tables are created in Table Editor

## 5. Test Connection
Run: `python scripts/test_supabase.py`
