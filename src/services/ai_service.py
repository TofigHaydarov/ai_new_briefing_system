import logging
import functools
from tenacity import retry, stop_after_attempt, wait_exponential

# Importing provided AI modules without modifying their public interface
from ai.llm import summarize_and_label
from ai.embedding import embed
from ai.schemas import Article, LabeledSummary

# Setup structured logging for this module
logger = logging.getLogger(__name__)

class AIIntegrationError(Exception):
    """Custom exception class for handling AI service failures gracefully."""
    pass

class AIService:
    @staticmethod
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def safe_summarize(article: Article) -> LabeledSummary:
        """
        Summarizes an article and assigns a topic. 
        Implements exponential backoff for transient API errors.
        """
        try:
            logger.debug(f"Starting summarization for URL: {article.url} | Title: {article.title}")
            result = summarize_and_label(article)
            logger.info(f"Successfully summarized article: {article.url} | Assigned Topic: {result.topic}")
            return result
        except Exception as e:
            logger.error(f"Summarization failed for URL {article.url}: {str(e)}")
            raise AIIntegrationError(f"LLM Summarization Error: {str(e)}") from e

    @staticmethod
    @functools.lru_cache(maxsize=1000)
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def safe_embed(text: str) -> list[float]:
        """
        Generates embeddings for a given text.
        Results are cached to prevent redundant API calls for identical text.
        """
        try:
            logger.debug(f"Generating embedding for text (length: {len(text)} chars)")
            result = embed(text)
            logger.info("Successfully generated embedding.")
            return result
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise AIIntegrationError(f"Embedding Error: {str(e)}") from e