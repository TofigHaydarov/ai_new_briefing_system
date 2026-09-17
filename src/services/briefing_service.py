from datetime import datetime, timezone
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.schemas import Article, Digest, DigestItem
from src.storage.repository import UserProfile
from src.services.ai_service import AIService, AIIntegrationError

async def generate_user_digest(
    profile: UserProfile, 
    articles: list[Article]
) -> Digest:
    
    digest_items: list[DigestItem] = []

    filtered_articles = [
        art for art in articles 
        if art.source.lower() not in [s.lower() for s in profile.excluded_sources]
    ]

    for article in filtered_articles:
        try:
            labeled_summary = AIService.safe_summarize(article)
            user_topics = [t.lower() for t in profile.preferred_topics]
            if "general" in user_topics or labeled_summary.topic.value.lower() in user_topics:
                digest_items.append(
                    DigestItem(article=article, labeled=labeled_summary)
                )

            topic_count = sum(
                1 for item in digest_items if item.labeled.topic == labeled_summary.topic
            )
            if topic_count >= profile.max_items_per_topic:
                continue
        except AIIntegrationError as err:
            print(f"Skipping article due to AI error: {err}")
            continue
        except Exception as err:
            print(f"[Warning] Failed to process article '{article.title}': {err}")
            continue

    
    return Digest(
        user=profile.user,
        generated_at=datetime.now(timezone.utc),
        items=digest_items,
    )