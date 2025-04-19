import urllib
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, element
from ebooklib import epub

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib.epub")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib.epub")

TEXT_TAGS = ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]


def flatten(t):
    return (
        [j for i in t for j in flatten(i)]
        if isinstance(t, tuple | list)
        else [t]
        if isinstance(t, epub.Link)
        else []
    )


@dataclass(eq=True, frozen=True)
class EpubParagraph:
    chapter: int
    element: element.Tag
    references: list

    def text(self):
        return "".join(self.element.strings)


@dataclass(eq=True, frozen=True)
class EpubChapter:
    content: BeautifulSoup
    title: str
    is_linear: bool
    idx: int

    def text(self):
        paragraphs = self.content.find("body").find_all(TEXT_TAGS)
        r = []
        for p in paragraphs:
            if "id" in p.attrs:
                continue
            r.append(EpubParagraph(chapter=self.idx, element=p, references=[]))
        return r


@dataclass(eq=True, frozen=True)
class Epub:
    epub: epub.EpubBook
    path: Path
    title: str
    chapters: list

    def text(self):
        return [p for c in self.chapters for p in c.text()]

    @classmethod
    def from_file(cls, path):
        file = epub.read_epub(path, {"ignore_ncx": True})

        flat_toc = flatten(file.toc)
        m = {
            it.id: i
            for i, e in enumerate(flat_toc)
            if (
                it := file.get_item_with_href(
                    urllib.parse.unquote(e.href.split("#")[0])
                )
            )
        }
        if len(m) != len(flat_toc):
            print(
                "WARNING: Couldn't fully map toc to chapters, contact the dev, preferably with the epub"
            )

        chapters = []
        prev_title = ""
        for i, v in enumerate(file.spine):
            item = file.get_item_with_id(v[0])
            title = flat_toc[m[v[0]]].title if v[0] in m else ""

            if item.media_type != "application/xhtml+xml":
                if title:
                    prev_title = title
                continue

            content = BeautifulSoup(item.get_content(), "html.parser")

            r = content.find("body").find_all(TEXT_TAGS)
            # Most of the time chapter names are on images
            idx = 0
            while idx < len(r) and not r[idx].get_text().strip():
                idx += 1
            if idx >= len(r):
                if title:
                    prev_title = title
                continue

            if not title:
                if t := prev_title.strip():
                    title = t
                    prev_title = ""
                elif len(t := r[idx].get_text().strip()) < 25:
                    title = t
                else:
                    title = item.get_name()

            chapter = EpubChapter(content=content, title=title, is_linear=v[1], idx=i)
            chapters.append(chapter)
        return cls(
            epub=file,
            path=path,
            title=file.title.strip() or path.name,
            chapters=chapters,
        )
