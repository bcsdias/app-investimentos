import streamlit as st
from supabase import create_client, Client
from cryptography.fernet import Fernet
from src.utils.logger import logger

def _get_cipher() -> Fernet:
    """
    Inicializa o objeto Fernet usando a chave de segurança definida no st.secrets.
    A chave deve ser compatível com Fernet (gerada via Fernet.generate_key()).
    """
    try:
        key = st.secrets["security"]["fernet_key"]
        return Fernet(key.encode())
    except Exception as e:
        logger.error(f"Erro ao inicializar Fernet cipher: {e}")
        raise ValueError("Chave de segurança 'security.fernet_key' não configurada ou inválida.")

def _get_supabase() -> Client:
    """
    Inicializa o cliente Supabase usando a URL e a Service Role Key do st.secrets.
    A Service Role Key é usada para permitir operações de backend que ignoram RLS 
    (conforme acordado para validação manual no backend).
    """
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_role_key"]
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Erro ao inicializar cliente Supabase: {e}")
        raise ValueError("Configurações do Supabase não encontradas no st.secrets.")

def save_dlp_token(user_email: str, raw_token: str):
    """
    Criptografa o token DLP e armazena/atualiza no Supabase para o usuário especificado.
    """
    try:
        cipher = _get_cipher()
        encrypted_token = cipher.encrypt(raw_token.encode()).decode()
        
        supabase = _get_supabase()
        data = {
            "user_email": user_email,
            "encrypted_token": encrypted_token
        }
        
        # O Supabase upsert usa a Primary Key (user_email) para decidir entre insert ou update
        response = supabase.table("user_tokens").upsert(data).execute()
        return response
    except Exception as e:
        logger.error(f"Erro ao salvar token para {user_email}: {e}")
        raise

def load_dlp_token(user_email: str) -> str:
    """
    Recupera e descriptografa o token DLP do usuário especificado.
    Retorna None se o token não for encontrado.
    """
    try:
        supabase = _get_supabase()
        response = supabase.table("user_tokens").select("encrypted_token").eq("user_email", user_email).execute()
        
        if not response.data:
            return None
            
        encrypted_token = response.data[0]["encrypted_token"]
        cipher = _get_cipher()
        decrypted_token = cipher.decrypt(encrypted_token.encode()).decode()
        
        return decrypted_token
    except Exception as e:
        logger.error(f"Erro ao carregar token para {user_email}: {e}")
        # Se falhar a descriptografia, pode ser que a chave mudou
        raise

def delete_dlp_token(user_email: str):
    """
    Remove o registro do token do usuário especificado do Supabase.
    """
    try:
        supabase = _get_supabase()
        response = supabase.table("user_tokens").delete().eq("user_email", user_email).execute()
        return response
    except Exception as e:
        logger.error(f"Erro ao deletar token para {user_email}: {e}")
        raise
