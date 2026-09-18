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

def _log_and_raise_final_error(retry_state):
    """Logs an ERROR only when all retries are exhausted, preventing log spam on every attempt."""
    exc = retry_state.outcome.exception()
    logger.error(f"Retries exhausted. Final AI Service error: {str(exc)}")
    raise exc

class AIService:
    # Cache dictionary for storing summaries based on article content hash
    _summary_cache = {}

    @staticmethod
    def safe_summarize(article: Article) -> LabeledSummary:
        """
        Summarizes an article and assigns a topic.
        Implements content-based caching to avoid redundant LLM calls.
        """
        c_hash = content_hash(article.content)
        
        if c_hash in AIService._summary_cache:
            logger.debug(f"Cache hit for article: {article.title}")
            return AIService._summary_cache[c_hash]

        result = AIService._execute_summarize(article)
        AIService._summary_cache[c_hash] = result
        return result

    @staticmethod
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Do not retry on ValueError (empty content) or ProviderError (schema/format issues)
        retry=retry_if_not_exception_type((ValueError, ProviderError)),
        retry_error_callback=_log_and_raise_final_error
    )
    def _execute_summarize(article: Article) -> LabeledSummary:
        """
        Internal method to execute summarization with exponential backoff.
        Does not retry on ValueError (e.g., empty content).
        """
        try:
            logger.debug(f"Starting summarization for URL: {article.url}")
            result = summarize_and_label(article)
            logger.info(f"Successfully summarized article: {article.url}")
            return result
        except (ValueError, ProviderError) as ve:
            # Raise the original exception without masking it to satisfy test requirements
            logger.error(f"Validation/Schema error for {article.url}: {str(ve)}")
            raise ve
        except Exception as e:
            # Log as WARNING to avoid spamming the error logs on intermediate retries.
            logger.warning(f"Summarization attempt failed (retrying...): {str(e)}")
            raise AIIntegrationError(f"LLM Summarization Error: {str(e)}") from e

    @staticmethod
    @functools.lru_cache(maxsize=1000)
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type((ValueError, ProviderError)),
        retry_error_callback=_log_and_raise_final_error
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
        except (ValueError, ProviderError) as ve:
            logger.error(f"Validation/Schema error during embedding: {str(ve)}")
            raise ve
        except Exception as e:
            # Log as WARNING to avoid spamming the error logs on intermediate retries.
            logger.warning(f"Embedding attempt failed (retrying...): {str(e)}")
            raise AIIntegrationError(f"Embedding Error: {str(e)}") from e