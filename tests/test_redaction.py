import json
import math

import pytest

from transcript_pipeline.redaction import REDACTED, redact_text, redact_value


def test_redact_text_removes_header_env_and_database_uri_secrets() -> None:
    bot_secret = "bot.fake.token-123"
    env_secret = "env-secret-456"
    db_user = "fake-db-user"
    db_password = "fake-db-password"
    db_query_secret = "query-secret-789"
    text = (
        "GET /api/v10/channels failed; "
        f"Authorization: Bot {bot_secret}\n"
        f"API_TOKEN={env_secret}\n"
        "database=postgresql://"
        f"{db_user}:{db_password}@cluster.example.test/pipeline"
        f"?authSource=admin&password={db_query_secret}"
    )

    cleaned = redact_text(text)

    for secret in (
        bot_secret,
        env_secret,
        db_user,
        db_password,
        db_query_secret,
    ):
        assert secret not in cleaned
    assert "GET /api/v10/channels failed" in cleaned
    assert "cluster.example.test/pipeline" in cleaned
    assert "authSource=" in cleaned
    assert REDACTED in cleaned


def test_redact_value_recurses_without_mutating_nested_input() -> None:
    header_secret = "nested-header-secret"
    exception_secret = "exception-env-secret"
    password_secret = "nested-password-secret"
    payload = {
        "request": {"Authorization": f"Bearer {header_secret}"},
        "events": [
            "message retrieval failed after page 3",
            RuntimeError(
                f"worker failed while API_TOKEN={exception_secret}; retry is safe"
            ),
            {"password": password_secret, "attempt": 3},
        ],
    }

    cleaned = redact_value(payload)
    rendered = json.dumps(cleaned)

    assert payload["request"]["Authorization"] == f"Bearer {header_secret}"
    for secret in (header_secret, exception_secret, password_secret):
        assert secret not in rendered
    assert "RuntimeError" in rendered
    assert "message retrieval failed after page 3" in rendered
    assert '"attempt": 3' in rendered


def test_redaction_handles_inline_quoted_token_assignments() -> None:
    secret = "quoted-env-token"

    cleaned = redact_text(f'command failed with API_TOKEN="{secret}" while starting')

    assert secret not in cleaned
    assert "while starting" in cleaned


def test_nested_database_uri_keeps_host_context_but_not_credentials() -> None:
    secret = "nested-db-secret"
    payload = {
        "database_uri": (
            f"postgresql://diagnostic-user:{secret}@db.example.test/pipeline"
            f"?password={secret}"
        )
    }

    rendered = json.dumps(redact_value(payload))

    assert secret not in rendered
    assert "diagnostic-user" not in rendered
    assert "db.example.test/pipeline" in rendered


def test_redaction_is_idempotent_across_repeated_manifest_writes() -> None:
    original = "Upstream API returned 401 for Authorization: Bot fake-token"

    once = redact_text(original)

    assert redact_text(once) == once


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Service token: fake-secret",
        "token=fake-secret",
    ],
)
def test_common_token_diagnostics_are_redacted(diagnostic: str) -> None:
    cleaned = redact_text(diagnostic)

    assert "fake-secret" not in cleaned
    assert REDACTED in cleaned


def test_exception_with_token_mapping_is_redacted_without_losing_context() -> None:
    error = RuntimeError({"token": "fake-secret", "context": "page 4"})

    cleaned = redact_value(error)

    assert "fake-secret" not in cleaned
    assert "RuntimeError" in cleaned
    assert "page 4" in cleaned


@pytest.mark.parametrize("sensitive_key", ["password", "credential", "client_secret"])
def test_exception_mapping_args_are_recursively_redacted(
    sensitive_key: str,
) -> None:
    secret = f"fake-{sensitive_key}-value"
    error = RuntimeError({sensitive_key: secret, "context": "page 4"})

    cleaned = redact_value(error)

    assert secret not in cleaned
    assert cleaned.startswith("RuntimeError: ")
    assert sensitive_key in cleaned
    assert REDACTED in cleaned
    assert "page 4" in cleaned
    assert redact_value(cleaned) == cleaned


def test_multi_argument_exception_redacts_nested_args_and_preserves_structure() -> None:
    password = "fake-password-value"
    credential = "fake-credential-value"
    error = RuntimeError(
        "worker request failed",
        {"password": password, "context": "page 4"},
        ["retryable", {"credential": credential}],
        503,
    )

    cleaned = redact_value(error)

    assert password not in cleaned
    assert credential not in cleaned
    assert cleaned.startswith("RuntimeError: (")
    assert "worker request failed" in cleaned
    assert "page 4" in cleaned
    assert "retryable" in cleaned
    assert "503" in cleaned
    assert cleaned.count(REDACTED) == 2


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ToKeN: fake-secret, retry on page 4",
        "API-KEY=fake-secret; retry on page 4",
        'api_key: "fake-secret" then retry on page 4',
        "AUTHORIZATION=Bearer fake-secret then retry on page 4",
        "url?access_token=fake-secret&context=page-4",
    ],
)
def test_keyed_secret_forms_are_case_insensitive_and_preserve_context(
    diagnostic: str,
) -> None:
    cleaned = redact_text(diagnostic)

    assert "fake-secret" not in cleaned
    assert "page 4" in cleaned.replace("page-4", "page 4")
    assert REDACTED in cleaned
    assert redact_text(cleaned) == cleaned


@pytest.mark.parametrize("sensitive_key", ["accessToken", "clientSecret", "apiKey"])
def test_camel_case_sensitive_mapping_keys_are_redacted(
    sensitive_key: str,
) -> None:
    secret = f"fake-{sensitive_key}-quality-gate-secret"

    cleaned = redact_value({sensitive_key: secret, "context": "page 4"})
    rendered = json.dumps(cleaned)

    assert secret not in rendered
    assert cleaned[sensitive_key] == REDACTED
    assert cleaned["context"] == "page 4"


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic fake-basic-credential",
        "authorization=BEARER fake-bearer-credential",
        "AUTHORIZATION: bot fake-bot-credential",
    ],
)
def test_authorization_schemes_redact_the_entire_credential(header: str) -> None:
    cleaned = redact_text(f"request failed; {header}, context page 4")

    assert "fake-" not in cleaned
    assert "Basic" not in cleaned
    assert "BEARER" not in cleaned
    assert " bot " not in cleaned
    assert "context page 4" in cleaned
    assert cleaned.count(REDACTED) == 1
    assert redact_text(cleaned) == cleaned


@pytest.mark.parametrize(
    "diagnostic",
    [
        "password=fake-password context page 4",
        "PASSWORD: fake-password, context page 4",
        "secret=fake-secret; context page 4",
        'clientSecret: "fake secret with spaces" then context page 4',
        "credential=fake-credential&context=page-4",
    ],
)
def test_free_form_sensitive_values_are_redacted_without_losing_context(
    diagnostic: str,
) -> None:
    cleaned = redact_text(diagnostic)

    assert "fake" not in cleaned
    assert "page 4" in cleaned.replace("page-4", "page 4")
    assert cleaned.count(REDACTED) == 1
    assert redact_text(cleaned) == cleaned


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected_recursively(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        redact_value({"nested": [value]})
    with pytest.raises(ValueError, match="non-finite"):
        redact_value({value: "non-finite mapping key"})


def test_sensitive_assignments_inside_diagnostic_strings_are_not_skipped() -> None:
    password = "nested-password-leak"
    authorization = "nested-basic-leak"
    diagnostic = (
        '{"details": "password='
        f'{password}", "failure": "wrapper: Authorization: Basic {authorization}", '
        '"context": "page 4"}'
    )

    cleaned = redact_text(diagnostic)

    assert password not in cleaned
    assert authorization not in cleaned
    assert "page 4" in cleaned
    assert cleaned.count(REDACTED) == 2


def test_non_sensitive_key_names_are_left_unchanged() -> None:
    diagnostic = "notsecret=value; monkey=banana; context page 4"

    assert redact_text(diagnostic) == diagnostic


def test_prefixed_api_key_header_is_redacted() -> None:
    cleaned = redact_text("X-API-Key: fake-api-secret, context page 4")

    assert "fake-api-secret" not in cleaned
    assert "context page 4" in cleaned
    assert REDACTED in cleaned


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "credentials",
        "tokens",
        "secrets",
        "passwords",
        "apiKeys",
        "clientSecrets",
        "accessTokens",
    ],
)
def test_plural_sensitive_mapping_keys_are_redacted(sensitive_key: str) -> None:
    secret = f"plural-{sensitive_key}-secret"

    cleaned = redact_value({sensitive_key: secret, "context": "page 4"})

    assert cleaned[sensitive_key] == REDACTED
    assert secret not in json.dumps(cleaned)
    assert cleaned["context"] == "page 4"


@pytest.mark.parametrize(
    "safe_key",
    ["token_count", "password_policy", "authorization_url"],
)
def test_noncredential_key_suffixes_are_not_false_positives(safe_key: str) -> None:
    value = "safe diagnostic value"

    assert redact_value({safe_key: value}) == {safe_key: value}
    assert redact_text(f"{safe_key}={value}") == f"{safe_key}={value}"


@pytest.mark.parametrize(
    ("uri", "safe_context"),
    [
        (
            "postgresql://db-user:db-password@db.example.test/pipeline"
            "?sslmode=require&password=query-secret",
            "postgresql://[REDACTED]@db.example.test/pipeline",
        ),
        (
            "https://api-user:api-password@api.example.test/v1/messages"
            "?apiKey=query-secret&context=page-4",
            "https://[REDACTED]@api.example.test/v1/messages",
        ),
        (
            "redis://cache-user:cache-password@cache.example.test:6379/0",
            "redis://[REDACTED]@cache.example.test:6379/0",
        ),
    ],
)
def test_common_uri_userinfo_and_query_values_are_redacted(
    uri: str, safe_context: str
) -> None:
    cleaned = redact_text(f"connection failed: {uri}")

    for secret in (
        "db-user",
        "db-password",
        "api-user",
        "api-password",
        "cache-user",
        "cache-password",
        "query-secret",
        "require",
        "page-4",
    ):
        assert secret not in cleaned
    assert safe_context in cleaned
    assert REDACTED in cleaned


def test_file_uris_and_ordinary_paths_are_not_treated_as_credentials() -> None:
    diagnostic = (
        "source file:///srv/media/session.mp4 and /srv/media/session.mp4 page 4"
    )

    assert redact_text(diagnostic) == diagnostic


def test_structured_json_diagnostic_redacts_escaped_password_as_a_unit() -> None:
    secret = 'json-password#with"quote'
    diagnostic = json.dumps(
        {"password": secret, "context": "page 4", "attempt": 2}
    )

    cleaned = redact_text(diagnostic)
    parsed = json.loads(cleaned)

    assert secret not in cleaned
    assert "json-password" not in cleaned
    assert "with" not in cleaned
    assert parsed == {"password": REDACTED, "context": "page 4", "attempt": 2}


def test_python_literal_diagnostic_is_safely_parsed_and_redacted() -> None:
    secret = "literal-password#with'quote"
    diagnostic = repr(
        {"clientSecrets": secret, "context": "page 4", "retryable": True}
    )

    cleaned = redact_text(diagnostic)
    parsed = json.loads(cleaned)

    assert secret not in cleaned
    assert parsed == {
        "clientSecrets": REDACTED,
        "context": "page 4",
        "retryable": True,
    }


def test_free_form_escaped_quoted_password_is_fully_redacted() -> None:
    diagnostic = 'password="part-one\\"part-two#part-three" context page 4'

    cleaned = redact_text(diagnostic)

    for fragment in ("part-one", "part-two", "part-three"):
        assert fragment not in cleaned
    assert "context page 4" in cleaned
    assert cleaned.count(REDACTED) == 1


def test_mapping_keys_are_normalized_to_safe_json_strings_before_redaction() -> None:
    secret = "bytes-key-secret"

    cleaned = redact_value(
        {
            b"clientSecrets": secret,
            4: "four",
            True: "boolean",
            None: "none",
        }
    )

    assert secret not in json.dumps(cleaned)
    assert cleaned["clientSecrets"] == REDACTED
    assert cleaned["4"] == "four"
    assert cleaned["true"] == "boolean"
    assert cleaned["null"] == "none"


@pytest.mark.parametrize("key", [("tuple", "key"), object(), b"\xff"])
def test_unsupported_mapping_keys_are_rejected(key: object) -> None:
    with pytest.raises(ValueError, match="mapping key"):
        redact_value({key: "value"})
