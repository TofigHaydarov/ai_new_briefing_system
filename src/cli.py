import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
from typing import List, Optional
import typer
import logging
from storage.repository import JSONUserRepo, UserProfile
from src.config import settings
from src.services.briefing_service import generate_user_digest
from ai.schemas import Article, Digest
from datetime import datetime

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cli")

app = typer.Typer(help="AI News Briefing System CLI")
repo = JSONUserRepo(settings.JSON_STORAGE_PATH)


@app.command(name="get-profile")
def show_profile(username: str = typer.Argument(..., help="Username to retrieve")):
    logger.info(f"Fetching profile for user: '{username}'")
    profile = asyncio.run(repo.get_profile(username))
    
    if not profile:
        logger.error(f"Profile for user '{username}' not found.")
        typer.echo(f"ERROR: Profile for user '{username}' not found.", err=True)
        raise typer.Exit(code=1)
        
    logger.debug(f"Retrieved profile payload: {profile}")
    typer.echo(str(profile))


@app.command(name="create-profile")
def add_user(
    username: str = typer.Option(..., "--user", "-u", help="Username / User ID"),
    topics: List[str] = typer.Option(["General"], "--topics", "-t", help="Preferred topics"),
    excluded: List[str] = typer.Option([], "--excluded", "-e", help="Excluded sources/domains"),
    max_items: int = typer.Option(3, "--max-items", "-m", help="Max articles per topic"),
):
    logger.info(f"Attempting to create profile for user: '{username}'")
    try:
        profile = UserProfile(
            user=username,
            preferred_topics=topics,
            excluded_sources=excluded,
            max_items_per_topic=max_items
        )
        asyncio.run(repo.save_profile(profile))
        logger.info(f"Profile successfully saved for user: '{username}'")
        typer.echo("Profile Saved Successfully!")
    except Exception as err:
        logger.error(f"Failed to create profile for '{username}': {err}")
        typer.echo(f"ERROR: Invalid parameters or failure saving profile: {err}", err=True)
        raise typer.Exit(code=1)


@app.command(name="list-users")
def list_users():
    logger.info("Listing all registered user profiles.")
    profiles = asyncio.run(repo.get_all_profiles())

    if not profiles:
        logger.warning("No user profiles found in storage repository.")
        typer.echo("No user profiles found in storage.")
        return

    logger.info(f"Successfully retrieved {len(profiles)} profile(s).")
    typer.echo(f"Found {len(profiles)} user profile(s):")
    for profile in profiles:
        typer.echo(f" - {profile.user} (Topics: {', '.join(profile.preferred_topics)})")


@app.command(name="delete-profile")
def delete_profile(
    username: str = typer.Option(..., "--user", "-u", help="Username to delete")
):
    logger.info(f"Attempting to delete profile for user: '{username}'")
    deleted = asyncio.run(repo.delete_profile(username))

    if not deleted:
        logger.error(f"Deletion failed: User '{username}' not found.")
        typer.echo(f"ERROR: Could not delete profile. User '{username}' not found.", err=True)
        raise typer.Exit(code=1)

    logger.info(f"Successfully deleted user profile: '{username}'")
    typer.echo(f"Successfully deleted profile for '{username}'.")
def _render_markdown_digest(digest: Digest) -> str:
    date_str = digest.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Daily AI News Digest for {digest.user}",
        f"_Generated on {date_str}_\n",
        "---",
    ]

    grouped = digest.by_topic()
    if not grouped:
        lines.append("\n*No articles match your preference profile today.*")

    for topic, items in grouped.items():
        lines.append(f"\n## Topic: {topic.value}\n")
        for item in items:
            lines.append(f"### {item.article.title}")
            lines.append(f"Source: {item.article.source} | Sentiment: {item.labeled.sentiment.value.capitalize()}")
            lines.append(f"URL: {item.article.url}\n")
            lines.append(f"{item.labeled.summary}\n")

    return "\n".join(lines)

@app.command(name="generate-briefing")
@app.command(name="run-daily")
def generate_briefing(
    username: Optional[str] = typer.Option(
        None, "--user", "-u", help="Target username for briefing generation"
    ),
    run_all: bool = typer.Option(
        False, "--all", "-a", help="Generate briefings for all registered users"
    ),
):
   
    if not username and not run_all:
        logger.error("Neither --user nor --all flag was provided.")
        typer.echo("ERROR: Please specify a user using --user <username> or pass --all.", err=True)
        raise typer.Exit(code=1)


    sample_articles = [
        Article(
            title="New AI Model Released",
            url="https://techcrunch.com/example",
            source="TechCrunch",
            content="A ground-breaking AI model was announced today capable of...",
        )
    ]

    async def _process_user(profile) -> None:
        logger.info(f"Processing briefing pipeline for '{profile.user}'...")
        digest = await generate_user_digest(profile, sample_articles)
        
        md_content = _render_markdown_digest(digest)
        
        date_prefix = digest.generated_at.strftime("%Y-%m-%d")
        settings.DIGEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = settings.DIGEST_OUTPUT_DIR / f"{date_prefix}-{profile.user}.md"
        
        output_file.write_text(md_content, encoding="utf-8")
        logger.info(f"Saved briefing for '{profile.user}' to {output_file}")
        typer.echo(f"  [+] Briefing generated for '{profile.user}': {output_file}")

    async def _runner():
        if run_all:
            profiles = await repo.get_all_profiles()
            if not profiles:
                typer.echo("No user profiles found.", err=True)
                return
            await asyncio.gather(*[_process_user(p) for p in profiles])
        else:
            profile = await repo.get_profile(username)
            if not profile:
                logger.error(f"User profile '{username}' not found.")
                typer.echo(f"ERROR: User '{username}' not found.", err=True)
                raise typer.Exit(code=1)
            await _process_user(profile)

    asyncio.run(_runner())


@app.command(name="get-briefing")
def get_briefing(
    username: str = typer.Option(..., "--user", "-u", help="Username to retrieve briefing for"),
    date_str: Optional[str] = typer.Option(None, "--date", "-d", help="Date in YYYY-MM-DD format (defaults to today)")
):
    target_date = date_str or datetime.now().strftime("%Y-%m-%d")
    digest_file = settings.DIGEST_OUTPUT_DIR / f"{target_date}-{username}.md"

    if not digest_file.exists():
        logger.warning(f"Digest file missing: {digest_file}")
        typer.echo(f"ERROR: No briefing digest found for '{username}' on date {target_date}.", err=True)
        raise typer.Exit(code=1)

    typer.echo(digest_file.read_text(encoding="utf-8"))
    
if __name__ == "__main__":
    app()