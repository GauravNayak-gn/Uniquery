import asyncio
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig,
    LXMLWebScrapingStrategy, BFSDeepCrawlStrategy,
    DefaultMarkdownGenerator, CacheMode, PruningContentFilter,
    FilterChain, ContentTypeFilter, DomainFilter
)

def strip_markdown_links(md: str) -> str:
    # Replace <!--citation:1--> -> text and remove bare URLs
    md = re.sub(r'\[([^\]]+)\]\((?:[^)]+)\)', r'\1', md)
    md = re.sub(r'(?<!\()https?://\S+', '', md)
    return md

def dedupe_lines(text: str) -> str:
    seen = set()
    out_lines = []
    for line in text.splitlines():
        # Normalize for de-dup comparison
        key = re.sub(r'\s+', ' ', line.strip().lower())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out_lines.append(line)
    # Collapse excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(out_lines)).strip()
    return cleaned

async def main():
    # Headless browser
    browser_config = BrowserConfig(headless=True)

    # Crawl only this domain; allow all internal paths (no fragile *.htm filters)
    filter_chain = FilterChain([
        DomainFilter(allowed_domains=["app1.unipune.ac.in"]),
        ContentTypeFilter(allowed_types=["text/html"]),
    ])

    # Keep social/external filtering in case pages contain embeds
    custom_excluded_domains = [
        "facebook.com","twitter.com","linkedin.com","instagram.com","youtube.com",
        "pinterest.com","reddit.com","tiktok.com","snapchat.com","whatsapp.com",
        "telegram.org","medium.com","tumblr.com","flickr.com","vimeo.com",
        "discord.com","discord.gg","vk.com","weibo.com","wechat.com"
    ]
    social_media_domains = [
        "facebook.com","fb.com","fb.me","twitter.com","t.co","x.com","instagram.com",
        "instagr.am","linkedin.com","lnkd.in","youtube.com","youtu.be","tiktok.com",
        "pinterest.com","pin.it","snapchat.com","reddit.com","redd.it","tumblr.com",
        "tmblr.co","whatsapp.com","wa.me","telegram.org","t.me","discord.com",
        "discord.gg","twitch.tv","vk.com","weibo.com","wechat.com"
    ]

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=3,              # go deeper to collect all internal pages
            include_external=False,   # stay on-domain
            filter_chain=filter_chain,
            max_pages=200,            # raise cap so we don't stop at 20
        ),
        # Only scrape inside div#content on each page
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={
                # We want clean text only
                "ignore_links": True,                 # drop link targets, keep anchor text
                "ignore_images": True,                # no images
                # Link/media filtering
                "exclude_external_links": True,
                "exclude_social_media_links": True,
                "exclude_domains": custom_excluded_domains,
                "exclude_social_media_domains": social_media_domains,
            }
        ),
        # Keep only meaningful content blocks
        word_count_threshold=40,
        excluded_tags=[
            # Navigation and UI
            "header","footer","nav","aside","menu","menuitem",
            # Scripts and styles
            "script","style","noscript",
            # Media/interactive
            "iframe","embed","object","applet",
            # Form elements
            "form","button","input","select","textarea","label","fieldset",
            # Ads/tracking
            "ins","advertisement",
            # Social embeds
            "twitter-widget","fb-like","fb-share-button",
            # Other non-content
            "svg","canvas","map","area",
            # IMPORTANT: do NOT exclude 'a' — we keep its text while ignoring hrefs
        ],
        # CSS scope: only extract from the content container
        css_selector="#content",   # was "content[id='content']" — fixed to target div#content
        cache_mode=CacheMode.ENABLED,
        verbose=True,
    )

    # Track cross-document duplicates by content hash
    seen_hashes = set()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(
            url="https://app1.unipune.ac.in/dkbhave/default.htm",
            config=run_config
        )

        rag_data = {
            "source": "Savitribai Phule Pune University - DK Bhave",
            "base_url": "https://app1.unipune.ac.in/dkbhave/default.htm",
            "timestamp": datetime.now().isoformat(),
            "documents": []
        }

        def add_doc(i, result, content):
            # Prefer engine-supplied title, else first H1 in markdown
            title = (getattr(result, "metadata", {}) or {}).get("title") or ""
            if not title:
                m = re.search(r"^\s*#\s+(.+)$", content, flags=re.MULTILINE)
                title = m.group(1).strip() if m else f"Page {i+1}"

            doc = {
                "id": f"doc_{i+1}",
                "url": result.url,
                "title": title,
                "content": content,
                "metadata": {
                    "word_count": len(content.split()),
                    "character_count": len(content),
                    "crawl_depth": (getattr(result, "metadata", {}) or {}).get("depth", 0),
                    "source_url": result.url,
                    "scraped_at": datetime.now().isoformat()
                }
            }
            rag_data["documents"].append(doc)

        def process_result(i, result):
            if not (getattr(result, "success", False) and getattr(result, "markdown", "")):
                print(f"✗ Page {i+1}: {getattr(result, 'url', 'unknown')} - Failed or empty")
                return

            # Basic cleanup
            md = result.markdown

            # Keep only content in div#content (extra guard if upstream selector missed)
            if "#content" not in run_config.css_selector:
                # Not expected, but left for safety
                pass

            # Remove markdown links, bare URLs and dedupe lines
            md = strip_markdown_links(md)
            md = dedupe_lines(md)

            # Skip tiny/boilerplate pages after cleanup
            if len(md.split()) < 40:
                print(f"⚠ Page {i+1}: {result.url} - Skipped (too little content after filtering)")
                return

            # Cross-document dedupe
            content_hash = hashlib.sha256(md.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                print(f"↩ Page {i+1}: {result.url} - Duplicate content, skipped")
                return
            seen_hashes.add(content_hash)

            add_doc(i, result, md)
            print(f"✓ Page {i+1}: {result.url}  ({len(md.split())} words)")

        if isinstance(results, list):
            if not results:
                print("No results crawled. Check URL or crawl constraints.")
                return

            print(f"Processing {len(results)} pages...\n")
            for i, res in enumerate(results):
                process_result(i, res)
        else:
            process_result(0, results)

        # Save to JSON file
        output_dir = Path("scraped_data")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = output_dir / f"unipune_dkbhave_{timestamp}_rag.json"

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)

        # Summary
        print("=" * 80)
        print("✓ Scraping Complete!")
        print("=" * 80)
        print(f"\n📁 Output saved to: {output_filename}")
        print("\n📊 Summary:")
        print(f"  • Total documents: {len(rag_data['documents'])}")
        if rag_data["documents"]:
            total_words = sum(d["metadata"]["word_count"] for d in rag_data["documents"])
            total_chars = sum(d["metadata"]["character_count"] for d in rag_data["documents"])
            avg_words = total_words // len(rag_data["documents"])
            print(f"  • Total word count: {total_words:,}")
            print(f"  • Total characters: {total_chars:,}")
            print(f"  • Average words per document: {avg_words:,}")
            print("\n📄 First few documents:")
            for d in rag_data["documents"][:5]:
                print(f"  • {d['title'][:60]}... ({d['metadata']['word_count']} words)")
            if len(rag_data["documents"]) > 5:
                print(f"  ... and {len(rag_data['documents']) - 5} more")

        print("\n" + "=" * 80)
        print("Ready for RAG pipeline processing!")
        print("=" * 80 + "\n")

        return output_filename

if __name__ == "__main__":
    output_file = asyncio.run(main())
    if output_file:
        print(f"💡 Next step: Load '{output_file}' in your chunking script for post-processing")