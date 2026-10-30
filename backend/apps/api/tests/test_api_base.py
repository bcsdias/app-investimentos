import json


def test_docs_served(client):
    assert client.get("/api/v1/docs").status_code == 200


def test_openapi_title(client):
    schema = client.get("/api/v1/openapi.json").json()
    assert schema["info"]["title"] == "app-investimentos API"


def test_handle_api_error_renders_contract(rf):
    from apps.api.errors import ApiError
    from config.api import handle_api_error

    resp = handle_api_error(rf.get("/"), ApiError(409, "dlp_token_missing", "Cadastre seu token DLP.", None))
    assert resp.status_code == 409
    assert json.loads(resp.content) == {
        "code": "dlp_token_missing",
        "message": "Cadastre seu token DLP.",
        "details": None,
    }
