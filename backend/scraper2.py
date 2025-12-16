import asyncio
import json
from datetime import datetime
from pathlib import Path
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, 
    LXMLWebScrapingStrategy, BFSDeepCrawlStrategy, 
    DefaultMarkdownGenerator, CacheMode, PruningContentFilter, 
    FilterChain, ContentTypeFilter, URLPatternFilter, DomainFilter
)

async def main():
    # Global browser config: Headless for efficiency
    browser_config = BrowserConfig(headless=True)

    # Create filter chain for internal HTM/HTML links only
    filter_chain = FilterChain([
        # Filter for HTM/HTML file extensions
        URLPatternFilter(
            patterns=[
                "*.htm",
                "*.html",
                "*/default.htm",
                "*/"  # Include directory URLs that might serve HTML
            ]
        ),
        # Filter for the specific domain only
        DomainFilter(
            allowed_domains=["app1.unipune.ac.in"],
            blocked_domains=[]
        ),
        # Filter by content type
        ContentTypeFilter(
            allowed_types=["text/html", "text/htm"]
        )
    ])

    # Define domains to exclude (customize as needed)
    custom_excluded_domains = [
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "instagram.com",
        "youtube.com",
        "pinterest.com",
        "reddit.com",
        "tiktok.com",
        "snapchat.com",
        "whatsapp.com",
        "telegram.org",
        "medium.com",
        "tumblr.com",
        "flickr.com",
        "vimeo.com",
        # Add any other domains you want to exclude
    ]

    # Extended social media domains list
    social_media_domains = [
        "facebook.com", "fb.com", "fb.me",
        "twitter.com", "t.co", "x.com",
        "instagram.com", "instagr.am",
        "linkedin.com", "lnkd.in",
        "youtube.com", "youtu.be",
        "tiktok.com",
        "pinterest.com", "pin.it",
        "snapchat.com",
        "reddit.com", "redd.it",
        "tumblr.com", "tmblr.co",
        "whatsapp.com", "wa.me",
        "telegram.org", "t.me",
        "discord.com", "discord.gg",
        "twitch.tv",
        "vk.com",
        "weibo.com",
        "wechat.com"
    ]

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False,
            filter_chain=filter_chain,
            max_pages=20,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={
                "ignore_links": True,  # Keep links to filter them
                "ignore_images": False,  # Keep images to filter them
                # Link filtering options
                "exclude_external_links": True,  # Remove external links
                "exclude_social_media_links": True,  # Remove social media links
                "exclude_domains": custom_excluded_domains,  # Custom domain exclusion
                "exclude_social_media_domains": social_media_domains,  # Extended social media list
                # Media filtering options
                "exclude_external_images": True,  # Remove external images
            }
        ),
        # Content filtering thresholds and exclusions
        word_count_threshold=50,  # Skip text blocks with fewer than 50 words
        excluded_tags=[
            # Navigation and UI elements
            'form', 'header', 'footer', 'nav', 'aside', 'menu', 'menuitem','a'
            # Scripts and styles
            'script', 'style', 'noscript',
            # Media and interactive elements
            'iframe', 'embed', 'object', 'applet',
            # Form elements
            'button', 'input', 'select', 'textarea', 'label', 'fieldset',
            # Advertisement and tracking
            'ins', 'adsense', 'advertisement',
            # Social media embeds
            'twitter-widget', 'fb-like', 'fb-share-button',
            # Other non-content elements
            'svg', 'canvas', 'map', 'area',
        ],
        # Additional exclusion patterns
        exclude_external_links=True,  # Double ensure external links are excluded
        exclude_social_media_links=True,  # Double ensure social media links are excluded
        css_selector="content[id='content']",
        
        cache_mode=CacheMode.ENABLED,
        verbose=True,
    )

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
        
        if isinstance(results, list):
            if not results:
                print("No results crawled. Check URL or config.")
                return
            
            print(f"Processing {len(results)} pages...\n")
            
            for i, result in enumerate(results):
                if result.success and result.markdown:
                    # Extract title from markdown if available
                    title = ""
                    if result.markdown:
                        lines = result.markdown.split('\n')
                        for line in lines:
                            if line.strip() and not line.startswith('*') and not line.startswith('#'):
                                title = line.strip()
                                break
                            elif line.startswith('#'):
                                title = line.strip('#').strip()
                                break
                    
                    # Only add documents with actual content
                    if result.markdown and len(result.markdown.strip()) > 0:
                        doc = {
                            "id": f"doc_{i+1}",
                            "url": result.url,
                            "title": title or f"Page {i+1}",
                            "content": result.markdown,
                            "metadata": {
                                "word_count": len(result.markdown.split()),
                                "character_count": len(result.markdown),
                                "crawl_depth": result.metadata.get('depth', 0),
                                "source_url": result.url,
                                "scraped_at": datetime.now().isoformat()
                            }
                        }
                        rag_data["documents"].append(doc)
                        
                        print(f"✓ Page {i+1}: {result.url}")
                        print(f"  Title: {doc['title']}")
                        print(f"  Words: {doc['metadata']['word_count']:,}")
                        print(f"  Characters: {doc['metadata']['character_count']:,}\n")
                    else:
                        print(f"⚠ Page {i+1}: {result.url} - No content found\n")
                else:
                    print(f"✗ Page {i+1}: {result.url} - Failed to crawl\n")
            
        else:
            # Single result
            if results.success and results.markdown:
                doc = {
                    "id": "doc_1",
                    "url": results.url,
                    "title": "Main Page",
                    "content": results.markdown,
                    "metadata": {
                        "word_count": len(results.markdown.split()),
                        "character_count": len(results.markdown),
                        "crawl_depth": 0,
                        "source_url": results.url,
                        "scraped_at": datetime.now().isoformat()
                    }
                }
                rag_data["documents"].append(doc)
            else:
                print(f"Failed to crawl: {results.error_message if hasattr(results, 'error_message') else 'Unknown error'}")
                return
        
        # Save to JSON file
        output_dir = Path("scraped_data")
        output_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = output_dir / f"unipune_dkbhave_{timestamp}_rag.json"
        
        # Save the RAG-ready JSON
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"{'='*80}")
        print(f"✓ Scraping Complete!")
        print(f"{'='*80}")
        print(f"\n📁 Output saved to: {output_filename}")
        print(f"\n📊 Summary:")
        print(f"  • Total documents: {len(rag_data['documents'])}")
        
        if rag_data['documents']:
            total_words = sum(doc['metadata']['word_count'] for doc in rag_data['documents'])
            total_chars = sum(doc['metadata']['character_count'] for doc in rag_data['documents'])
            avg_words = total_words // len(rag_data['documents'])
            
            print(f"  • Total word count: {total_words:,}")
            print(f"  • Total characters: {total_chars:,}")
            print(f"  • Average words per document: {avg_words:,}")
            print(f"\n📄 Documents scraped:")
            for doc in rag_data['documents'][:5]:  # Show first 5
                print(f"  • {doc['title'][:50]}... ({doc['metadata']['word_count']} words)")
            if len(rag_data['documents']) > 5:
                print(f"  ... and {len(rag_data['documents'])-5} more documents")
        
        print(f"\n{'='*80}")
        print("Ready for RAG pipeline processing!")
        print(f"{'='*80}\n")
        
        return output_filename

if __name__ == "__main__":
    output_file = asyncio.run(main())
    
    if output_file:
        print(f"💡 Next step: Load '{output_file}' in your chunking script for post-processing")