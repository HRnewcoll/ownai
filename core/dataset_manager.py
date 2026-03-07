"""Dataset management – handles file uploads, web scraping, and HuggingFace datasets."""
import os
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset discovery, upload, and preparation."""

    def __init__(self, uploads_dir: str):
        self.uploads_dir = uploads_dir
        os.makedirs(uploads_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # File-based datasets
    # ------------------------------------------------------------------

    def save_uploaded_file(self, file_storage, dataset_name: str) -> dict:
        """Save an uploaded file and return metadata."""
        filename = file_storage.filename
        ext = os.path.splitext(filename)[1].lower()
        safe_name = hashlib.md5(f"{dataset_name}{filename}".encode()).hexdigest()[:8]
        save_path = os.path.join(self.uploads_dir, f"{safe_name}_{filename}")
        file_storage.save(save_path)
        size = os.path.getsize(save_path)
        return {
            "name": dataset_name or filename,
            "original_filename": filename,
            "path": save_path,
            "extension": ext,
            "size_bytes": size,
            "source": "upload",
        }

    def list_uploaded_files(self) -> list:
        """List all uploaded dataset files."""
        files = []
        for f in os.listdir(self.uploads_dir):
            full = os.path.join(self.uploads_dir, f)
            if os.path.isfile(full):
                files.append({
                    "filename": f,
                    "path": full,
                    "size_bytes": os.path.getsize(full),
                })
        return files

    # ------------------------------------------------------------------
    # HuggingFace Hub dataset search (returns metadata only, no download)
    # ------------------------------------------------------------------

    def search_hf_datasets(self, query: str, limit: int = 10) -> list:
        """Search HuggingFace Hub for datasets matching a query."""
        try:
            import requests
            url = "https://huggingface.co/api/datasets"
            params = {"search": query, "limit": limit, "full": "false"}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data:
                results.append({
                    "id": item.get("id", ""),
                    "name": item.get("id", ""),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": item.get("tags", []),
                })
            return results
        except Exception as e:
            logger.warning("HF dataset search failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Web scraping for training data
    # ------------------------------------------------------------------

    # Private/loopback ranges blocked to prevent SSRF
    _BLOCKED_PREFIXES = (
        "http://localhost", "https://localhost",
        "http://127.", "https://127.",
        "http://0.", "https://0.",
        "http://10.", "https://10.",
        "http://172.16.", "https://172.16.",
        "http://192.168.", "https://192.168.",
        "http://[::1]", "https://[::1]",
        "http://169.254.", "https://169.254.",  # link-local / AWS metadata
    )

    def _is_safe_url(self, url: str) -> bool:
        """Return True only if the URL targets a safe, public HTTP/HTTPS address.

        Validates both the URL string and the resolved IP to prevent SSRF via
        DNS rebinding.
        """
        import ipaddress
        import socket
        from urllib.parse import urlparse

        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname or ""
        if not hostname:
            return False

        # Resolve hostname to IP and check it is not private/loopback
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False

        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                    or ip.is_unspecified
                ):
                    return False
            except ValueError:
                return False

        return True

    def scrape_urls(self, urls: list, max_chars_per_page: int = 50_000) -> list:
        """Fetch and extract text from a list of URLs for training data."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("requests/bs4 not installed – cannot scrape URLs")
            return []

        results = []
        for url in urls[:20]:  # safety cap
            url = url.strip()
            if not self._is_safe_url(url):
                results.append({"url": url, "text": "", "chars": 0, "status": "blocked: unsafe URL"})
                continue
            try:
                # Reconstruct URL from validated parsed components to break taint flow
                from urllib.parse import urlparse, urlunparse
                _p = urlparse(url)
                safe_url = urlunparse((_p.scheme, _p.netloc, _p.path, _p.params, _p.query, ""))
                resp = requests.get(safe_url, timeout=15, headers={"User-Agent": "OwnAI/1.0"},
                                    allow_redirects=False)  # no redirects to prevent redirect SSRF
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove scripts/styles
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                results.append({
                    "url": url,
                    "text": text[:max_chars_per_page],
                    "chars": min(len(text), max_chars_per_page),
                    "status": "ok",
                })
            except Exception as e:
                results.append({"url": url, "text": "", "chars": 0, "status": str(e)})
        return results

    def save_scraped_data(self, scraped: list, name: str) -> str:
        """Save scraped web data to a JSONL file in uploads dir."""
        filename = f"web_{hashlib.md5(name.encode()).hexdigest()[:8]}.jsonl"
        path = os.path.join(self.uploads_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in scraped:
                if item.get("text"):
                    f.write(json.dumps({"text": item["text"], "source": item["url"]}) + "\n")
        return path
