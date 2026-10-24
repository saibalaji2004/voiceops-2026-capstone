"""Step 3 conversational ZentroMart support agent with memory and mock tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.rag_chatbot import respond_with_rag
from Capstone_Project_Dataset_FTTT_2026.tools.tools import (
    create_ticket,
    lookup_order,
)


ORDER_ID_AFTER_ORDER = re.compile(
    r"\borders?\s*(?:id|number|no\.?\s*)?\s*(?:is\s+|[:#]\s*)?"
    r"((?:ORD[- ]?)?\d{4,12})\b",
    flags=re.IGNORECASE,
)

ORDER_ID_STANDALONE = re.compile(
    r"\b(ORD[- ]?\d{4,12})\b",
    re.IGNORECASE,
)

# Used only when the agent has explicitly asked for an order ID.
BARE_ORDER_ID = re.compile(
    r"(?:ORD[- ]?)?(\d{4,12})",
    flags=re.IGNORECASE,
)

ORDER_TASK = re.compile(
    r"\b(?:my\s+order|order\s+(?:status|details|tracking|number|id)|"
    r"track(?:ing)?\s+(?:my\s+)?order|where(?:'s|\s+is)?\s+(?:my\s+)?order|"
    r"order.*(?:status|track|deliver|arriv|ship|cancel|refund|return|damage|broken|late|delay|missing|lost)|"
    r"(?:status|track|deliver|arriv|ship|cancel|refund|return|damage|broken|late|delay|missing|lost).*order|"
    r"never\s+placed|did(?:n't|\s+not)\s+place\s+(?:this\s+)?order)\b|"
    r"मेरा\s+ऑर्डर|ऑर्डर.*(?:स्थिति|स्टेटस|डिलीवर|रद्द|रिफंड|ट्रैक)",
    flags=re.IGNORECASE,
)

ORDER_STATUS_ONLY = re.compile(
    r"\b(?:status|track|tracking|where|arrival|arrive|deliver|delivery|"
    r"shipped|late|delay)\b|(?:स्टेटस|स्थिति|कहाँ|कब.*डिलीवर)",
    re.IGNORECASE,
)

FOLLOW_UP = re.compile(
    r"\b(?:it|that|this|they|them|those|same|then|also|what about|"
    r"how long|and if|why so)\b|(?:उसका|उसकी|उसे|वह|फिर|कितने दिन|"
    r"कब तक|और अगर)",
    re.IGNORECASE,
)

VERIFY_DIGITS = re.compile(r"(?<!\d)(\d{6}|\d{4})(?!\d)")

EXPLICIT_HANDOVER = re.compile(
    r"\b(?:human(?:\s+(?:agent|support|representative))?|live agent|"
    r"real person|talk to (?:a person|someone|an agent)|connect me|"
    r"create (?:a )?ticket|raise (?:a )?complaint|escalate|supervisor)\b|"
    r"मानव सहायता|एजेंट से बात|शिकायत दर्ज|टिकट बनाइए|सपोर्ट से बात",
    re.IGNORECASE,
)

ANSWER_HANDOVER = re.compile(
    r"\b(?:human support|human agent|support colleague|"
    r"escalate(?:d|ion)?|hand over|handover|"
    r"contact (?:a human|human support|customer support)|"
    r"connect (?:you|you with) (?:to )?(?:a )?(?:human|support)|"
    r"needs? human review|requires? human review)\b|"
    r"मानव सहायता|मानव एजेंट|सपोर्ट टीम|मानव समीक्षा",
    re.IGNORECASE,
)


@dataclass
class AgentResponse:
    """Structured result returned by one conversational turn."""

    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    action: str = "rag"
    tool_result: dict[str, Any] | None = None


class Step3Agent:
    """Conversational agent with bounded memory and verification-aware tools."""

    def __init__(self, top_k: int = 8, memory_messages: int = 12):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.top_k = top_k
        self.memory_messages = max(4, memory_messages)

        self.history: list[dict[str, str]] = []
        self.introduced = False

        # Important: this state is used by respond() to recognize a bare ID.
        self.awaiting_order_id = False

        self.pending_order_id: str | None = None
        self.verification_attempts = 0
        self.locked_order_ids: set[str] = set()
        self.verified_orders: dict[str, dict[str, Any]] = {}
        self.last_verified_order_id: str | None = None
        self.ticketed_messages: dict[str, str] = {}

    @staticmethod
    def _has_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text))

    def _intro(self, user_text: str) -> str:
        if self.introduced:
            return ""

        self.introduced = True

        if self._has_devanagari(user_text):
            return (
                "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ।\n\n"
            )

        return "Hello, I’m ZentroMart’s AI customer-support assistant.\n\n"

    def _finish(
        self,
        user_text: str,
        answer: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        action: str = "rag",
        tool_result: dict[str, Any] | None = None,
        history_user_text: str | None = None,
    ) -> AgentResponse:
        """Build the reply and retain a bounded, privacy-aware history."""

        answer = self._intro(user_text) + answer.strip()

        stored_user_text = (
            history_user_text
            if history_user_text is not None
            else user_text
        )

        self.history.extend([
            {"role": "user", "content": stored_user_text},
            {"role": "assistant", "content": answer},
        ])

        if len(self.history) > self.memory_messages:
            self.history = self.history[-self.memory_messages:]

        return AgentResponse(
            text=answer,
            sources=sources or [],
            provider=provider,
            model=model,
            action=action,
            tool_result=tool_result,
        )

    @staticmethod
    def _digits_only(value: str) -> str:
        """Keep only ASCII digits for IDs and verification values."""

        return "".join(
            ch for ch in str(value)
            if ch.isascii() and ch.isdigit()
        )

    def _extract_order_id(self, message: str) -> str | None:
        """Extract IDs mentioned with an order or prefixed with ORD."""

        match = (
            ORDER_ID_AFTER_ORDER.search(message)
            or ORDER_ID_STANDALONE.search(message)
        )

        if not match:
            return None

        digits = self._digits_only(match.group(1))
        return digits or None

    @staticmethod
    def _is_order_request(message: str) -> bool:
        return bool(ORDER_TASK.search(message))

    def _order_summary(self, order: dict[str, Any]) -> str:
        """Provide a minimum order summary after successful verification."""

        order_id = str(order.get("order_id", ""))
        status = order.get("status") or "Not available in the order record"

        lines = [
            f"I verified the caller and found order {order_id}.",
            f"Current status: {status}.",
        ]

        promised = order.get("promised_date")
        delivered = order.get("delivered_on")

        if promised:
            lines.append(f"Promised delivery date: {promised}.")

        if delivered:
            lines.append(f"Recorded delivery date: {delivered}.")

        return_status = order.get("return_status")
        if return_status:
            lines.append(f"Return status: {return_status}.")

        refund_initiated = (
            order.get("return_refund_initiated_on")
            or order.get("cancellation_refund_initiated_on")
            or order.get("rto_refund_initiated_on")
        )

        refund_completed = order.get("return_refund_completed_on")

        if refund_initiated:
            lines.append(
                f"Refund initiation date recorded: {refund_initiated}."
            )

        if refund_completed:
            lines.append(
                f"Refund completion date recorded: {refund_completed}."
            )

        lines.append(
            "I can use this verified order context for your follow-up "
            "questions in this conversation."
        )

        return "\n".join(lines)

    def _handle_pending_verification(
        self,
        user_text: str,
    ) -> AgentResponse:
        """Verify the caller, allowing one retry and never storing the proof."""

        order_id = self.pending_order_id

        if order_id is None:
            raise RuntimeError("Verification state is inconsistent.")

        digits = VERIFY_DIGITS.findall(user_text)

        if not digits:
            return self._finish(
                user_text,
                "For verification, enter only the registered phone "
                "number’s last 4 digits or the order’s 6-digit delivery "
                "pincode. Do not send an OTP, password, or payment details.",
                action="verification_required",
                history_user_text=(
                    "[Customer was asked to provide verification digits; "
                    "no digits stored.]"
                ),
            )

        # Prefer the six-digit delivery pincode if the user entered one.
        proof = next(
            (digit for digit in digits if len(digit) == 6),
            digits[0],
        )

        self.verification_attempts += 1

        result = lookup_order(
            order_id,
            verification_value=proof,
        )

        if result.get("verified") is True and result.get("found") is True:
            self.verified_orders[order_id] = result
            self.last_verified_order_id = order_id
            self.pending_order_id = None
            self.verification_attempts = 0

            return self._finish(
                user_text,
                self._order_summary(result),
                action="lookup_order_verified",
                tool_result={
                    "verified": True,
                    "order_id": order_id,
                },
                history_user_text=(
                    "[Caller verification provided; value not retained.]"
                ),
            )

        # One retry is allowed after the initial failed verification.
        if self.verification_attempts < 2:
            return self._finish(
                user_text,
                "I couldn’t verify those details. The order information "
                "remains protected. You may try once more using the "
                "registered phone’s last 4 digits or the 6-digit delivery "
                "pincode.",
                action="verification_failed_retry_allowed",
                tool_result={
                    "verified": False,
                    "attempt": self.verification_attempts,
                    "retries_remaining": 1,
                },
                history_user_text=(
                    "[Caller verification failed; value not retained.]"
                ),
            )

        self.locked_order_ids.add(order_id)
        self.pending_order_id = None
        self.verification_attempts = 0

        return self._finish(
            user_text,
            "I couldn’t verify the details after two attempts, so I can’t "
            "disclose the order information in this conversation. Please "
            "use ZentroMart’s official support channel for further identity "
            "verification.",
            action="verification_locked",
            tool_result={
                "verified": False,
                "locked_for_conversation": True,
            },
            history_user_text=(
                "[Caller verification failed; value not retained.]"
            ),
        )

    def _create_handover(
        self,
        user_text: str,
        answer: str,
        *,
        reason: str,
        sources: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentResponse:
        """Create a ticket and claim success only after confirmation."""

        dedupe_key = re.sub(
            r"\s+",
            " ",
            user_text.casefold(),
        ).strip()

        previous_ticket = self.ticketed_messages.get(dedupe_key)

        if previous_ticket:
            ticket_suffix = (
                f"A support ticket ({previous_ticket}) was already "
                "created for this same message."
            )

            return self._finish(
                user_text,
                f"{answer.strip()}\n\n{ticket_suffix}",
                sources=sources,
                provider=provider,
                model=model,
                action="handover_ticket_reused",
                tool_result={
                    "created": True,
                    "ticket_id": previous_ticket,
                    "reused": True,
                },
            )

        context = "\n".join(
            f"{item['role']}: {item['content'][:500]}"
            for item in self.history[-6:]
        )

        summary = (
            f"Customer issue: {user_text.strip()}\n"
            f"Conversation context: {context or 'No earlier turns.'}\n"
            f"Assistant response/checks: {answer.strip()[:700]}\n"
            f"Escalation reason: {reason}. "
            "A human support colleague must review the issue."
        )

        verified_order = self.verified_orders.get(
            self.last_verified_order_id or ""
        )

        ticket = create_ticket(
            summary=summary,
            order_id=(
                str(verified_order.get("order_id"))
                if verified_order
                else None
            ),
            reason=reason,
            intent="human_handover",
        )

        if ticket.get("created") is True:
            ticket_id = str(ticket["ticket_id"])
            self.ticketed_messages[dedupe_key] = ticket_id

            final_answer = (
                f"{answer.strip()}\n\n"
                f"I created support ticket {ticket_id} for human follow-up. "
                "This confirms ticket creation, not that a human has "
                "already joined the conversation."
            )

            return self._finish(
                user_text,
                final_answer,
                sources=sources,
                provider=provider,
                model=model,
                action="create_ticket_success",
                tool_result={
                    "created": True,
                    "ticket_id": ticket_id,
                },
            )

        final_answer = (
            f"{answer.strip()}\n\n"
            "I couldn’t create the handover ticket in the support system, "
            "so no ticket was created. Please contact ZentroMart support "
            "directly."
        )

        return self._finish(
            user_text,
            final_answer,
            sources=sources,
            provider=provider,
            model=model,
            action="create_ticket_failed",
            tool_result={
                "created": False,
                "error": ticket.get("error", "Unknown error"),
            },
        )

    def respond(self, user_text: str) -> AgentResponse:
        """Process one customer turn using memory, tools, and RAG."""

        message = (user_text or "").strip()

        if not message:
            return AgentResponse(
                "Please enter a message.",
                action="empty_input",
            )

        # Verification has priority over all other input.
        # Verification values are never written to conversation history.
        if self.pending_order_id:
            return self._handle_pending_verification(message)

        # FIX: remember whether we explicitly asked the customer for an ID.
        was_awaiting_order_id = self.awaiting_order_id

        order_id = self._extract_order_id(message)

        # FIX: accept a bare numeric ID only when we are waiting for it.
        # This accepts 69520867 after "Please share your order ID".
        if was_awaiting_order_id and order_id is None:
            bare_match = BARE_ORDER_ID.fullmatch(message)

            if bare_match:
                order_id = self._digits_only(bare_match.group(1))
            else:
                return self._finish(
                    message,
                    "Please enter your order ID, using the ID shown "
                    "in your order confirmation.",
                    action="clarify_order_id",
                )

        # The next message is treated as an order request when the
        # preceding assistant turn explicitly asked for an order ID.
        is_order_request = (
            self._is_order_request(message)
            or was_awaiting_order_id
        )

        if order_id:
            self.awaiting_order_id = False

        if is_order_request:
            if order_id:
                if order_id in self.locked_order_ids:
                    return self._finish(
                        message,
                        "I can’t disclose details for that order after the "
                        "previous verification limit was reached. Please "
                        "use ZentroMart’s official support channel.",
                        action="verification_locked",
                    )

                if order_id in self.verified_orders:
                    self.last_verified_order_id = order_id

                    # Return stored status only for a simple status query.
                    if (
                        ORDER_STATUS_ONLY.search(message)
                        and not re.search(
                            r"\b(?:cancel|refund|return|damage|broken|"
                            r"compensation|policy|eligible|can i)\b",
                            message,
                            re.IGNORECASE,
                        )
                    ):
                        return self._finish(
                            message,
                            self._order_summary(
                                self.verified_orders[order_id]
                            ),
                            action="lookup_order_cached",
                            tool_result={
                                "verified": True,
                                "order_id": order_id,
                                "cached": True,
                            },
                        )

                else:
                    # Calling with the order ID alone does not expose
                    # whether the order exists or reveal its contents.
                    result = lookup_order(order_id)

                    if result.get("verification_required"):
                        self.pending_order_id = order_id
                        self.verification_attempts = 0

                        return self._finish(
                            message,
                            "I can check the order after verifying the "
                            "caller. Please provide either the registered "
                            "phone number’s last 4 digits or the 6-digit "
                            "delivery pincode. Do not share an OTP, "
                            "password, or payment credentials.",
                            action="verification_required",
                            tool_result={
                                "verification_required": True,
                            },
                        )

            elif self.last_verified_order_id:
                # Use the previously verified order for follow-ups.
                order = self.verified_orders.get(
                    self.last_verified_order_id
                )

                if (
                    order
                    and ORDER_STATUS_ONLY.search(message)
                    and not re.search(
                        r"\b(?:cancel|refund|return|damage|broken|"
                        r"compensation|policy|eligible|can i)\b",
                        message,
                        re.IGNORECASE,
                    )
                ):
                    return self._finish(
                        message,
                        self._order_summary(order),
                        action="lookup_order_cached",
                        tool_result={
                            "verified": True,
                            "order_id": self.last_verified_order_id,
                            "cached": True,
                        },
                    )
                # Otherwise the verified order remains available as context.

            else:
                # FIX: set state before asking, so the next numeric reply
                # is interpreted as the requested order ID, not a new query.
                self.awaiting_order_id = True

                return self._finish(
                    message,
                    "I can help check the order. Please share the order "
                    "ID first; I’ll then verify the caller before showing "
                    "any order details.",
                    action="clarify_order_id",
                )

        # Explicit requests for a person or a ticket use the real mock tool.
        if EXPLICIT_HANDOVER.search(message):
            return self._create_handover(
                message,
                "I’ll submit a handover request to the human support team.",
                reason=(
                    "Customer explicitly requested human support or a ticket"
                ),
            )

        # Add recent memory for likely follow-up questions.
        rag_question = message

        if self.history and FOLLOW_UP.search(message):
            recent = self.history[-6:]
            context = "\n".join(
                f"{item['role']}: {item['content'][:450]}"
                for item in recent
            )

            rag_question = (
                "Use this recent conversation only to resolve follow-up "
                "references; use retrieved knowledge-base evidence for "
                "policy facts.\nRecent conversation:\n"
                f"{context}\nCurrent customer follow-up: {message}"
            )

        verified_order = self.verified_orders.get(
            self.last_verified_order_id or ""
        )

        if verified_order and FOLLOW_UP.search(message):
            order_context = self._order_summary(verified_order)

            rag_question = (
                f"Verified order context from this conversation:\n"
                f"{order_context}\n\n"
                f"Customer conversation context:\n{rag_question}"
            )

        result = respond_with_rag(
            rag_question,
            top_k=self.top_k,
        )

        answer = str(
            result.get("answer")
            or "I couldn’t produce a response. Please try again."
        ).strip()

        sources = list(result.get("sources") or [])
        provider = result.get("provider")
        model = result.get("model")

        # If the RAG response indicates that a human is required,
        # create a ticket and confirm only what the tool reports.
        if ANSWER_HANDOVER.search(answer):
            return self._create_handover(
                message,
                answer,
                reason=(
                    "The RAG response indicates human review or support "
                    "is required"
                ),
                sources=sources,
                provider=provider,
                model=model,
            )

        return self._finish(
            message,
            answer,
            sources=sources,
            provider=provider,
            model=model,
            action="rag",
        )