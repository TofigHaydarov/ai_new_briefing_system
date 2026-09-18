import logging
import functools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

# Importing provided AI modules without modifying their public interface
from ai.llm import summarize_and_label
from ai.embedding import embed
from ai.schemas import Article, LabeledSummary
from config import settings
from ai.dedup import content_hash

# Setup structured logging for this module
logger = logging.getLogger(__name__)

class AIIntegrationError(Exception):
    """Custom exception class for handling AI service failures gracefully."""
    pass

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
        retry=retry_if_not_exception_type(ValueError),  # Do not retry on ValueError, as it indicates a bad request
        reraise=True
    )
    def _execute_summarize(article: Article) -> LabeledSummary:
        """
        Internal method to execute summarization with exponential backoff.
        Does not retry on ValueError (e.g., empty content).
        """
        try:
            logger.debug(f"Starting summarization for URL: {article.url} | Title: {article.title}")
            result = summarize_and_label(article)
            logger.info(f"Successfully summarized article: {article.url} | Assigned Topic: {result.topic}")
            return result
        except ValueError as ve:
            logger.error(f"Validation error for {article.url}: {str(ve)}")
            raise ve
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
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise AIIntegrationError(f"Embedding Error: {str(e)}") from e