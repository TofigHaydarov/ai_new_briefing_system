import logging
# Importing provided AI helper functions and schemas
from ai.dedup import url_canonicalize, content_hash, near_duplicate
from ai.schemas import Article

# Setup structured logging
logger = logging.getLogger(__name__)

class ArticleDeduplicator:
    def __init__(self, semantic_threshold: float = 0.85):
        """
        A two-stage deduplication system for articles.
        semantic_threshold: Threshold for semantic similarity. 
        (0.85 is selected as optimal for project reporting to balance minor text edits and core meaning).
        """
        self.semantic_threshold = semantic_threshold
        self.seen_urls = set()
        self.seen_hashes = set()
        self.accepted_articles = []

    def is_duplicate(self, article: Article) -> bool:
        """
        Checks if a single article is a duplicate based on a two-pass approach.
        """
        # PASS 1: Cheap pass (Fast check via URL and Hash)
        canon_url = url_canonicalize(article.url)
        if canon_url in self.seen_urls:
            logger.debug(f"Duplicate found (URL match): {canon_url}")
            return True
            
        c_hash = content_hash(article.text)
        if c_hash in self.seen_hashes:
            logger.debug(f"Duplicate found (Hash match): {article.title}")
            return True

        # PASS 2: Semantic pass (Contextual similarity check)
        for accepted in self.accepted_articles:
            if near_duplicate(article.text, accepted.text, threshold=self.semantic_threshold):
                logger.debug(f"Semantic duplicate found: '{article.title}' is similar to '{accepted.title}'")
                return True

        # If it passes all filters, it is unique. Add to memory.
        self.seen_urls.add(canon_url)
        self.seen_hashes.add(c_hash)
        self.accepted_articles.append(article)
        
        return False

    def process_batch(self, articles: list[Article]) -> list[Article]:
        """
        Filters a list of articles, returning only the unique ones.
        """
        logger.info(f"Deduplication started: evaluating {len(articles)} articles.")
        unique_articles = []
        
        for article in articles:
            if not self.is_duplicate(article):
                unique_articles.append(article)
        
        logger.info(f"Deduplication completed: retained {len(unique_articles)} unique articles.")
        return unique_articles