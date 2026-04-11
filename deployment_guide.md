# Guia de Configuração e Deploy (Fase 2)

Este guia fornece as instruções passo a passo para configurar os serviços externos necessários para o funcionamento do Dashboard de Investimentos v3.

## 1. Google Cloud Console (OAuth2)
Necessário para habilitar o login via `st.login()`.

1.  Acesse o [Google Cloud Console](https://console.cloud.google.com/).
2.  Crie um novo projeto ou selecione um existente.
3.  Vá em **APIs e Serviços > Tela de permissão OAuth**:
    *   Escolha "Externo" e preencha as informações obrigatórias.
    *   Adicione o escopo `.../auth/userinfo.email` e `.../auth/userinfo.profile`.
4.  Vá em **Credenciais > Criar credenciais > ID do cliente OAuth**:
    *   Tipo de aplicativo: **Aplicativo da Web**.
    *   **Origens JavaScript autorizadas**: `http://localhost:8501`.
    *   **URIs de redirecionamento autorizados**: `http://localhost:8501/oauth2callback`.
5.  Copie o **ID do cliente** e a **Chave secreta do cliente** para o seu `secrets.toml`.

> [!IMPORTANT]
> Se você fizer deploy para produção (ex: Streamlit Cloud), adicione a URL da sua aplicação nas Origens e URIs de redirecionamento (sempre terminando em `/oauth2callback`).

---

## 2. Supabase (Banco de Dados e Tokens)
Usado para armazenar as chaves DLP dos usuários de forma criptografada.

1.  Crie um projeto no [Supabase](https://supabase.com/).
2.  No menu lateral, vá em **SQL Editor**.
3.  Clique em "New Query" e cole o conteúdo do arquivo [supabase_schema.sql](file:///c:/onedrive-bcsdias/OneDrive/dev/app_investimentos/src/data/supabase_schema.sql). Execute a query.
4.  Vá em **Project Settings > API**:
    *   Copie a **Project URL**.
    *   Copie a **anon public key**.
    *   Copie a **service_role key** (necessária para o backend gerenciar os tokens).
5.  Cole essas chaves na seção `[supabase]` do seu `secrets.toml`.

---

## 3. Upstash Redis (Cache de Nuvem)
Usado para acelerar o carregamento de dados e reduzir requisições às APIs.

1.  Crie uma conta no [Upstash](https://upstash.com/).
2.  Crie uma nova base de dados **Redis**.
3.  Na aba "Details", procure pela seção **REST API**.
4.  Copie a **UPSTASH_REDIS_REST_URL** e o **UPSTASH_REDIS_REST_TOKEN**.
5.  Cole essas informações na seção `[upstash_redis]` do seu `secrets.toml`.

---

## 4. Chave de Criptografia (Security)
Para garantir que os tokens DLP fiquem ilegíveis no banco de dados.

1.  Você precisa gerar uma chave **Fernet** válida.
2.  Execute o seguinte comando no seu terminal/Python:
    ```python
    from cryptography.fernet import Fernet
    print(Fernet.generate_key().decode())
    ```
3.  Copie o valor gerado e cole em `fernet_key` na seção `[security]` do seu `secrets.toml`.

---

## 5. Resumo de Segredos (`secrets.toml`)
Seu arquivo final deve seguir o padrão do [.streamlit/secrets.toml.example](file:///c:/onedrive-bcsdias/OneDrive/dev/app_investimentos/.streamlit/secrets.toml.example).
