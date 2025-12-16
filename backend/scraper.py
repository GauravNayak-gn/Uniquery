import asyncio
import json
import re
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler

# =============================================================
# 🔹 Helpers
# =============================================================

URL_REGEX = re.compile(
    r"""(?xi)
    \b(
        (?:https?://|www\.)        # protocol or www
        [^\s<>()"'{}[\]]+          # url body
        (?:\([^\s<>()]*\)[^\s<>()]*)*
    )
    """
)

def normalize_url(base_domain: str, url: str) -> str:
    """Normalize a URL for de-duplication and consistent crawling."""
    absolute = urljoin(base_domain, url)
    parsed = urlparse(absolute)
    # Drop fragments; keep query (it may denote distinct pages on legacy sites)
    normalized = parsed._replace(fragment="")
    return urlunparse(normalized)

def determine_page_type(url):
    """Determine the page type based on URL patterns."""
    url_lower = url.lower()
    if 'faq' in url_lower:
        return 'FAQ'
    elif 'contact' in url_lower:
        return 'Contact'
    elif 'instruction' in url_lower or 'instructions' in url_lower:
        return 'Instructions'
    elif 'apply' in url_lower or 'newuser' in url_lower or 'admission' in url_lower:
        return 'Application'
    elif 'alumini' in url_lower or 'alumni' in url_lower:
        return 'Alumni'
    elif 'login' in url_lower or 'signin' in url_lower:
        return 'Login'
    elif 'default' in url_lower or url_lower.endswith('/'):
        return 'Home'
    else:
        return 'General'

def clean_text_content(text: str) -> str:
    """Normalize whitespace and strip artifacts."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_content_container(soup: BeautifulSoup):
    """
    Return the best content container element.
    Priority: #content -> main -> #main -> .content -> article -> body fallback
    """
    selectors = [
        "#content",
        "main",
        "#main",
        "[role='main']",
        ".content",
        "article",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup  # conservative fallback

def remove_nav_boilerplate(container: BeautifulSoup):
    """Remove non-content elements from the container."""
    for tag in container.find_all([
        "script", "style", "noscript", "iframe", "svg",
        "header", "footer", "nav", "aside", "form"
    ]):
        tag.decompose()

def strip_links(container: BeautifulSoup, keep_anchor_text=True, remove_bare_urls=True):
    """
    Remove all links:
    - If keep_anchor_text: unwrap <a> but keep its text.
    - Remove naked URLs (e.g., https://..., www...) from final text if remove_bare_urls.
    """
    for a in container.find_all("a"):
        if keep_anchor_text:
            # Replace the anchor with its text content (space-joined)
            a.replace_with(a.get_text(" ", strip=True))
        else:
            a.decompose()

    if remove_bare_urls:
        # Remove autolinked URLs that may appear as plain text
        for text_node in list(container.find_all(string=URL_REGEX)):
            new_text = URL_REGEX.sub("", text_node)
            text_node.replace_with(new_text)

def extract_visible_text_from_html(html: str) -> str:
    """
    Extract visible text from the best content container, preferring div#content.
    - Removes boilerplate.
    - Removes all links (tags and bare URLs).
    - Returns normalized text.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    container = extract_content_container(soup)  # prioritizes #content
    if not container:
        return ""

    remove_nav_boilerplate(container)
    strip_links(container, keep_anchor_text=True, remove_bare_urls=True)

    # Get text with paragraph-friendly separators
    text = container.get_text(separator="\n", strip=True)
    # Normalize whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return clean_text_content(text)

# =============================================================
# 🔹 Main Scraper
# =============================================================

async def scrape_website_for_rag(base_url: str, max_pages: int = 500):
    """
    Scrape website content including internal links,
    extract content from #content (with fallbacks),
    and store clean text for RAG.
    """
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    scraped_data = {
        "base_url": base_url,
        "scraped_timestamp": "",
        "total_pages": 0,
        "pages": []
    }

    visited_urls = set()
    urls_to_visit = [normalize_url(base_domain, base_url)]

    async with AsyncWebCrawler(verbose=True) as crawler:
        while urls_to_visit and len(visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue

            print(f"\n{'='*60}")
            print(f"Crawling: {current_url}")
            print(f"{'='*60}")

            try:
                result = await crawler.arun(
                    url=current_url,
                    word_count_threshold=10,
                    excluded_tags=['form', 'header', 'footer', 'nav'],
                    bypass_cache=True
                )

                if not result.success:
                    print(f"✗ Failed to scrape: {current_url}")
                    print(f"  Error: {getattr(result, 'error_message', 'Unknown error')}")
                    continue

                visited_urls.add(current_url)

                # Extract internal links to continue crawling
                internal_links = set()
                if getattr(result, "links", None):
                    for link_info in result.links.get('internal', []):
                        link_url = link_info.get('href', '')
                        if not link_url:
                            continue
                        absolute_url = normalize_url(base_domain, urljoin(current_url, link_url))
                        parsed_link = urlparse(absolute_url)
                        if parsed_link.netloc == urlparse(base_domain).netloc:
                            if absolute_url not in visited_urls and absolute_url not in urls_to_visit:
                                internal_links.add(absolute_url)
                                urls_to_visit.append(absolute_url)

                # ✅ Clean, link-free text from div#content (fallback-aware)
                clean_text = extract_visible_text_from_html(result.html)

                page_type = determine_page_type(current_url)
                title = (result.metadata.get('title', 'No Title')
                         if getattr(result, "metadata", None) else 'No Title')

                page_data = {
                    "page_id": f"page_{len(scraped_data['pages']) + 1}",
                    "url": current_url,
                    "title": title,
                    "content": clean_text,
                    "word_count": len(clean_text.split()) if clean_text else 0,
                    "metadata": {
                        "scraped_at": datetime.now().isoformat(),
                        "source_domain": urlparse(base_domain).netloc,
                        "page_type": page_type,
                        "content_length": len(clean_text)
                    }
                }

                scraped_data["pages"].append(page_data)

                print(f"✓ Scraped: {current_url}")
                print(f"  - Type: {page_type}")
                print(f"  - Internal links queued: {len(internal_links)}")
                print(f"  - Words: {page_data['word_count']}")
                if clean_text:
                    print(f"  - Preview: {clean_text[:140]}...")

            except Exception as e:
                print(f"✗ Exception while crawling {current_url}: {str(e)}")
                continue

    # Final summary
    scraped_data["scraped_timestamp"] = datetime.now().isoformat()
    scraped_data["total_pages"] = len(scraped_data["pages"])

    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total pages scraped: {len(scraped_data['pages'])}")
    print(f"Total URLs visited: {len(visited_urls)}")

    return scraped_data

# =============================================================
# 🔹 Entry Point
# =============================================================

async def main():
    url = "https://app1.unipune.ac.in/dkbhave/default.htm"

    print("Starting Universal Web Scraping for RAG...")
    print(f"Target: {url}\n")

    data = await scrape_website_for_rag(url, max_pages=500)

    output_file = "scraped_data_for_rag.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Data saved to {output_file}")
    print(f"✓ Pages scraped: {data['total_pages']}")
    print(f"✓ Total words: {sum(page['word_count'] for page in data['pages'])}")

    # Page type breakdown
    page_types = {}
    for page in data["pages"]:
        ptype = page["metadata"]["page_type"]
        page_types[ptype] = page_types.get(ptype, 0) + 1

    print(f"\nPage Types Summary:")
    for ptype, count in page_types.items():
        print(f"  - {ptype}: {count}")

if __name__ == "__main__":
    asyncio.run(main())