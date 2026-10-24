"""ZentroMart prompt-only chatbot using equal-peer LLM routing."""

from app.llm_router import generate_reply


SYSTEM_PROMPT = """
You are ZentroMart's customer support AI assistant.

At the start of a new conversation, clearly identify yourself as an AI assistant.
Be polite, helpful, concise, and answer the customer's actual question directly.

This is Step 1 of the VoiceOps capstone. You do not yet have ZentroMart's
company documents or connected support tools. This is a prompt-only baseline.

Do not automatically refuse a question just because it concerns a company policy.
Give your best-effort answer using general knowledge, and be transparent when a
company-specific fact cannot be verified. Never present an unverified guess as
a confirmed ZentroMart policy.

Never claim that you checked an order, verified a customer, consulted company
documents, or created a ticket unless the relevant functionality was actually used.
Never ask customers for passwords, OTPs, or payment credentials.

Respond in the same language as the customer where practical, including English,
Hindi, and Hinglish.

If you cannot reliably answer, acknowledge the uncertainty and recommend contacting
ZentroMart support.
""".strip()


class ZentroChat:
    """Maintain conversation context between user turns."""

    def __init__(self):
        self.messages = []

    def send_message(self, user_text: str):
        pending_messages = [
            *self.messages,
            {
                "role": "user",
                "content": user_text,
            },
        ]

        result = generate_reply(
            pending_messages,
            SYSTEM_PROMPT,
        )

        # Only store the turn after one provider answers successfully.
        self.messages = [
            *pending_messages,
            {
                "role": "assistant",
                "content": result.text,
            },
        ]

        return result


def create_chat() -> ZentroChat:
    return ZentroChat()