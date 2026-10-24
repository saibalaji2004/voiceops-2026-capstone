"""Test configured LLM endpoints, with optional provider filtering."""

import argparse

from app.llm_router import _call_provider, _provider_list


SYSTEM_PROMPT = (
    "You are testing API connectivity. "
    "Reply with exactly: OK"
)

MESSAGES = [
    {"role": "user", "content": "Reply with exactly OK."}
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test configured LLM model endpoints."
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "openrouter", "cloudflare"],
        help="Test only this provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Test only the first N matching endpoints. 0 means all.",
    )
    args = parser.parse_args()

    providers = _provider_list()

    if args.provider:
        providers = [
            provider for provider in providers
            if provider.name == args.provider
        ]

    if args.limit > 0:
        providers = providers[:args.limit]

    if not providers:
        print("No matching endpoints are configured.")
        return

    passed = 0
    failed = 0

    print(f"Testing {len(providers)} endpoint(s)\n")

    for number, provider in enumerate(providers, start=1):
        print(
            f"[{number}/{len(providers)}] "
            f"{provider.name} | {provider.model}"
        )

        try:
            text, actual_model = _call_provider(
                provider,
                MESSAGES,
                SYSTEM_PROMPT,
            )
            passed += 1
            print(f"  PASS | Actual model: {actual_model}")
            print(f"  Response: {text[:100].replace(chr(10), ' ')}")

        except Exception as exc:
            failed += 1
            print(
                f"  FAIL | {type(exc).__name__}: "
                f"{str(exc)[:250]}"
            )

        print()

    print("=" * 55)
    print(f"Endpoints tested: {len(providers)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 55)


if __name__ == "__main__":
    main()