import asyncio
from datetime import datetime, timezone
import logging
from typing import Sequence
from ai.dedup import url_canonicalize, content_hash, near_duplicate
from ai.schemas import Article, Digest, DigestItem, Topic
from src.services.ai_service import AIService, AIIntegrationError
from src.storage.repository import UserProfile

logger = logging.getLogger(__name__)


async def generate_user_digest(
    profile: UserProfile, 
    articles: Sequence[Article]
) -> Digest:
    digest_items: list[DigestItem] = []
    topic_counts: dict[Topic, int] = {t: 0 for t in Topic}
    
    seen_canonical_urls: set[str] = set()
    seen_content_hashes: set[str] = set()
    seen_embeddings: list[list[float]] = []

    user_preferred_topics = {t.lower() for t in profile.preferred_topics}
    excluded_sources = {s.lower() for s in profile.excluded_sources}

    for article in articles:
        
        canonical_url = url_canonicalize(article.url)
        c_hash = content_hash(article.content)

        if canonical_url in seen_canonical_urls or c_hash in seen_content_hashes:
            logger.info(f"Skipping exact duplicate (URL/Hash match): {article.title}")
            continue

        if article.source.lower() in excluded_sources:
            logger.debug(f"Skipping excluded source '{article.source}': {article.title}")
            continue

       
        try:
            curr_embedding = await asyncio.to_thread(AIService.safe_embed, article.content)
            
            is_near_dup = False
            for prev_embedding in seen_embeddings:
                if near_duplicate(curr_embedding, prev_embedding):
                    is_near_dup = True
                    break

            if is_near_dup:
                logger.info(f"Skipping near-duplicate (Semantic match): {article.title}")
                continue

        except AIIntegrationError as err:
            logger.warning(f"Embedding failed for '{article.title}', skipping dedup check: {err}")
            
        try:
            labeled_summary = await asyncio.to_thread(AIService.safe_summarize, article)
        except AIIntegrationError as err:
            logger.warning(f"Failed to summarize article '{article.title}': {err}")
            continue

        topic = labeled_summary.topic
        topic_name = topic.value.lower()

        is_preferred = "general" in user_preferred_topics or topic_name in user_preferred_topics
        if not is_preferred:
            continue

        if topic_counts[topic] >= profile.max_items_per_topic:
            logger.debug(f"Topic '{topic.value}' reached limit ({profile.max_items_per_topic}). Skipping.")
            continue

        seen_canonical_urls.add(canonical_url)
        seen_content_hashes.add(c_hash)
        if 'curr_embedding' in locals() and curr_embedding:
            seen_embeddings.append(curr_embedding)

        digest_items.append(DigestItem(article=article, labeled=labeled_summary))
        topic_counts[topic] += 1

    logger.info(f"Generated digest for user '{profile.user}' with {len(digest_items)} items.")
    return Digest(
        user=profile.user,
        generated_at=datetime.now(timezone.utc),
        items=digest_items,
    )