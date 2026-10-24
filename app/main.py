"""Terminal interface for the Step 3 ZentroMart conversational agent."""

from __future__ import annotations

from app.llm_router import AllProvidersUnavailable
from app.step3_agent import Step3Agent


def main() -> None:
    print("=" * 58)
    print("ZentroMart Customer Support | Step 3 conversational agent")
    print(
        "AI assistant with knowledge-grounded answers, "
        "order verification, and human handover."
    )
    print(
        "Order details require phone-last-4 or "
        "delivery-pincode verification."
    )
    print("Type 'exit' to end the conversation.")
    print("=" * 58)

    agent = Step3Agent(top_k=8)

    while True:
        try:
            user_text = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if user_text.casefold() in {"exit", "quit"}:
            print("Session ended.")
            break

        if not user_text:
            continue

        try:
            result = agent.respond(user_text)

            label = ""
            if result.provider and result.model:
                label = f" [{result.provider} | {result.model}]"

            print(f"\nZentroMart AI{label}:\n{result.text}")

            if result.sources:
                ids = [
                    str(source.get("section_id", "")).strip()
                    for source in result.sources
                ]
                ids = list(dict.fromkeys(item for item in ids if item))

                if ids:
                    print(
                        "Sources:",
                        ", ".join(f"[{item}]" for item in ids),
                    )

            if result.action and result.action != "rag":
                print(f"[Action: {result.action}]")

        except AllProvidersUnavailable as exc:
            print("\nNo configured LLM provider could answer right now.")
            print(str(exc))

        except Exception as exc:
            print(
                f"\nUnexpected error: "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()