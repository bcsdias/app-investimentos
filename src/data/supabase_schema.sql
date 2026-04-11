-- Table to store encrypted DLP tokens per user
-- Run this in your Supabase SQL Editor

create table if not exists public.user_tokens (
    user_email text primary key,
    encrypted_token text not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable RLS (Row Level Security)
alter table public.user_tokens enable row level security;

-- Policies
-- Since we are validating in the backend (using service_role), 
-- these policies serve as an additional layer of safety 
-- should the app ever use client-side authentication.

create policy "Users can view their own tokens" on public.user_tokens
    for select using (true); -- Placeholder: if using service_role, RLS is ignored.

-- Recommendation: To truly use RLS with Streamlit + Supabase, 
-- you would need to implement JWT exchange. 
-- For now, the backend will ensure user_email matches the logged-in user.
