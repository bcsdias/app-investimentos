def test_schema_error_uses_api_error_envelope(auth_client):
    r = auth_client.post(
        "/api/v1/portfolio/summary",
        data={"start": "não-data"},
        content_type="application/json",
    )
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "invalid_request"
    assert body["message"] == "Requisição inválida."
    assert "details" in body
    assert isinstance(body["details"], list)
