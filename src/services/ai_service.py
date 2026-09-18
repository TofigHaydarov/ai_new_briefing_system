import asyncio
import logging
import functools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

# Importing provided AI modules without modifying their public interface
from ai.llm import summarize_and_label
from ai.embedding import embed
from ai.schemas import Article, LabeledSummary
from ai.dedup import content_hash
from ai.providers.base import ProviderError

# Setup structured logging for this module
logger = logging.getLogger(__name__)


class AIIntegrationError(Exception):
    """Custom exception class for handling AI service failures gracefully."""
    pass

def wrap_in_ai_error(retry_state):
    """
    Called by Tenacity when retries are exhausted or stopped by the exception filter.
    Wraps the final exception in AIIntegrationError to satisfy test requirements.
    """
    exc = retry_state.outcome.exception()
    if isinstance(exc, AIIntegrationError):
        raise exc
    logger.error(f"Retries exhausted. Final AI Service error: {str(exc)}")
    raise AIIntegrationError(f"AI Service Error: {str(exc)}") from exc

class AIService:
    _summary_cache = {}

    @staticmethod
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(AIIntegrationError),
        retry_error_callback=wrap_in_ai_error
    )
    def safe_summarize(article: Article) -> LabeledSummary:
        """
        Summarizes an article and assigns a topic.
        Implements content-based caching to avoid redundant LLM calls.
        """
        # Incorporating the id() of the function prevents cache bleed across Pytest mock runs
        cache_key = f"{content_hash(article.content)}_{id(summarize_and_label)}"
        
        if cache_key in AIService._summary_cache:
            logger.debug(f"Cache hit for article: {article.title}")
            return AIService._summary_cache[cache_key]

        try:
            logger.debug(f"Starting summarization for URL: {article.url}")
            result = summarize_and_label(article)
            AIService._summary_cache[cache_key] = result
            logger.info(f"Successfully summarized article: {article.url}")
            return result
        except ValueError as ve:
            # Test expects AIIntegrationError immediately without retrying on ValueError
            logger.error(f"Validation error (empty content) for {article.url}: {str(ve)}")
            raise AIIntegrationError(f"Validation Error: {str(ve)}") from ve

    @staticmethod
    @functools.lru_cache(maxsize=1000)
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(AIIntegrationError),
        retry_error_callback=wrap_in_ai_error
    )
    def safe_embed(text: str) -> list[float]:
        """
        Generates embeddings for a given text.
        Results are cached to prevent redundant API calls for identical text.
        Returns a plain list of floats, converting from numpy arrays if necessary.
        """
        try:
            logger.debug(f"Generating embedding for text (length: {len(text)} chars)")
            result = embed(text)
            logger.info("Successfully generated embedding.")
            
            # Convert numpy array to list to match expected return type
            if hasattr(result, "tolist"):
                return result.tolist()
            return result
        except ValueError as ve:
            # Test expects AIIntegrationError immediately without retrying
            logger.error(f"Validation error (empty text) during embedding: {str(ve)}")
            raise AIIntegrationError(f"Validation Error: {str(ve)}") from ve