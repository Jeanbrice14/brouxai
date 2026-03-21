from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# v0 : notification structlog uniquement
# v1+ : email / Slack via SendGrid + Slack Webhooks


async def notify_hitl_required(
    report_id: str,
    checkpoint: str,
    tenant_id: str,
) -> None:
    """Notifie qu'une validation humaine est requise.

    v0 — log structuré uniquement.
    v1+ — email / Slack selon plan tarifaire du tenant.
    """
    logger.info(
        "hitl_required",
        report_id=report_id,
        checkpoint=checkpoint,
        tenant_id=tenant_id,
        review_url=f"/review/{report_id}?checkpoint={checkpoint}",
    )
